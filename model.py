from re import M
import torch
import torch.nn as nn
from data.config import epoch_to_drop
import torch.nn.functional as F
import os
import numpy as np
from models.alpha.operation import OPS
from data.config import PRIMITIVES
from data.config import cfg_nasmodel as cfg
#* concate two inner cell experiment



#info Cell_conv actually is innerCell
class InnerCell(nn.Module):
    #todo make it general def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell, alphas)
    def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell = PRIMITIVES):
        super(InnerCell, self).__init__()
        self.cellArchPerIneerCell = cellArchPerIneerCell
        self.numOfInnerCell = 2
        #Todo OPS[primitive]
        #todo maybe make it a dictionary layer_i_j_convName
        #info make operations to a list according cellArchPerIneerCell
        self.opList = nn.ModuleList()
        for opName in self.cellArchPerIneerCell:
            self.opList.append(OPS[opName](inputChannel, outputChannel, stride, False, False))
        
    def forward(self, input, alphasOfinnerCell):
        #info add each output of operation element-wise
        # print("next(model.parameters()).is_cuda", next(self.parameters()).is_cuda)
        out = self.opList[0](input)
        print(out.shape)
        for alpha, op in zip(alphasOfinnerCell, self.opList):
            #! Can NOT use inplace operation +=. WHY? 
            #! Ans: inplace operation make computational graphq fail
            print(out.shape, op(input).shape)
            out = out + alpha * op(input)
            
        return out


class Layer(nn.Module):
    def __init__(self, numOfInnerCell, layer, inputChannel=3, outputChannel=96, stride=1, padding=1,  cell=InnerCell):
        super(Layer, self).__init__()
        #info set private attribute
        self.numOfInnerCell = numOfInnerCell
        self.layer = layer
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        #info set inner cell
        self.innerCellList = nn.ModuleDict({
            'innerCell_'+str(layer)+'_0': cell(inputChannel, outputChannel//self.numOfInnerCell, stride),
            # 'innerCell_'+str(layer)+'_1': cell(inputChannel, outputChannel//self.numOfInnerCell, stride),
        })
    def forward(self, input, alphas):
        #* concate innerCell's output instead of add them elementwise
        indexOfInnerCell = 0
        output = 0
        for name in self.innerCellList:
            # add each inner cell directly without alphas involved
            if indexOfInnerCell == 0:
                output = self.innerCellList[name](input, alphas[indexOfInnerCell])
            else:
                output = torch.cat( (output, self.innerCellList[name](input, alphas[indexOfInnerCell])), dim=1 )

            indexOfInnerCell = indexOfInnerCell + 1
            # print("innerCellList{} output".format(name), output)
        return output



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #info private attribute
        self.numOfOpPerCell = cfg["numOfOperations"]
        self.numOfLayer = cfg["numOfLayers"]
        self.numOfInnerCell = cfg["numOfInnerCell"]
        self.alphasSaveDir = r'./alpha_pdart_nodrop'
        self.currentEpoch = 0
        self.maskSaveDir = r'./saved_mask_per_epoch'
        #info network structure
        self.layerDict = nn.ModuleDict({
            "layer_0":Layer(self.numOfInnerCell, 0, 3, 96, 4),
            # "layer_1":Layer(self.numOfInnerCell, 1, 96*numOfInnerCell, 96, 4),
            "layer_1":Layer(self.numOfInnerCell, 1, 96, 256, 1),
            "layer_2":Layer(self.numOfInnerCell, 2, 256, 384, 1),
            "layer_3":Layer(self.numOfInnerCell, 3, 384, 384, 1),
            "layer_4":Layer(self.numOfInnerCell, 4, 384, 256, 1),
            # "layer_6":Layer(self.numOfInnerCell, 6, 384*numOfInnerCell, 384, 1),
            # "layer_7":Layer(self.numOfInnerCell, 7, 384*numOfInnerCell, 384, 1),
            # "layer_8":Layer(self.numOfInnerCell, 8, 384*numOfInnerCell, 256, 1),
            # "layer_9":Layer(self.numOfInnerCell, 9, 256*numOfInnerCell, 256, 1),
            "max_pool1":nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            "max_pool2": nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            'max_pool3': nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        })
        # print(self.layerDict["layer_0"])
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, cfg["numOfClasses"])
        )
        
        #info initailize alphas and alphas mask, and register them with model
        self.initailizeAlphas()
        self.alphasMask = torch.full(self._alphas.shape, False, dtype=torch.bool) #* True means that element are masked(dropped)
        
        #* self.alphas can get the attribute
    def initailizeAlphas(self):
        #* set the first innercell's alpha to being evenly distributed, set second innercell's alphas to random
        self._alphas = F.softmax( 0.01*torch.ones([self.numOfLayer, self.numOfInnerCell, self.numOfOpPerCell], requires_grad=False), dim=-1 )
        tmp = F.softmax( torch.rand(self.numOfLayer, self.numOfInnerCell, self.numOfOpPerCell), dim=-1 )
        
        # #* set probability of innerCell with index 1 to random
        # for layer in range(self.numOfLayer):
        #     self._alphas[layer][1] = tmp[layer][1]
        
        self._alphas.requires_grad_(True)
        self.register_parameter("alphas", nn.Parameter(self._alphas))
        
    def dropMinAlpha(self):
        #* drop min alphas at certain epoch
        #* algo: find min for each innerCell and then modify mask
        (_, allMinIndex) = torch.topk(self.alphas, self.numOfOpPerCell, largest=False)
        for layer in range(self.numOfLayer):
            for indexOfInnerCell in range(self.numOfInnerCell):
                minIndexList = allMinIndex[layer][indexOfInnerCell].tolist() #* sorted index from small to large
                # minIndex = minIndexList.pop(0) #* pop the min index from a list
                for minIndex in minIndexList:
                    if self.alphasMask[layer][indexOfInnerCell][minIndex] == False:
                        self.alphasMask[layer][indexOfInnerCell][minIndex] = True
                        break
        print("dropMinAlpha mask", self.alphasMask)
    
    def saveAlphas(self, epoch, kthSeed):
        # print("save alphas")
        if not os.path.exists(self.alphasSaveDir):
            os.makedirs(self.alphasSaveDir)
        tmp = self.alphas.clone().detach()
        tmp = tmp.data.cpu().numpy()
        fileName =  os.path.join(self.alphasSaveDir, 'alpha_prob_' + str(kthSeed) + '_' + str(epoch))
        np.save(fileName, tmp)
        # print("\nepcho:", epoch, "save alphas:", tmp)
    def saveMask(self, epoch, kthSeed):
        # print("save mask")
        if not os.path.exists(self.maskSaveDir):
            os.makedirs(self.maskSaveDir)
        tmp = self.alphasMask.clone().detach()
        tmp = tmp.data.cpu().numpy()
        fileName =  os.path.join(self.maskSaveDir, 'mask_' + str(kthSeed) + '_' + str(epoch))
        np.save(fileName, tmp)
    
    def normalizeAlphas(self):
        #* set alphas to zero whose mask is true and pass it through softmax
        tmp = self.alphas.clone().detach()

        with torch.no_grad():
            #*inplace operation
            self.alphas *= 0
            self.alphas += F.softmax(tmp, dim=-1)
            # self.alphas= tmp #! this may cause optimizer will update old object
            # self.alphas = torch.nn.Parameter(tmp)
        # print("normalize alphas", self.alphas)
    def normalizeByDivideSum(self):
        #* set alphas to zero whose mask is true and pass it through softmax
        print("normalizeByDivideSum")
        # print("self.alphas", self.alphas, self.alphas.grad)
        
        tmp = self.alphas.clone().detach()

        with torch.no_grad():
            #*inplace operation
            self.alphas *= 0
            
            sum = torch.sum(tmp, dim=-1)
            sum = sum.reshape(tmp.size(dim=0), tmp.size(dim=1), -1)
            # print("sum", sum)
            self.alphas += tmp/sum # it will auto expand to match the size to alphas
        
        # print("self.alphas", self.alphas)
        # print("self.getAlphas()", self.getAlphas())
        
    def filtAlphas(self):
        #* set alphas to zero whose mask is true
        # tmp = self.alphas.clone().detach()
        with torch.no_grad():
            #*inplace operation
            self.alphas[self.alphasMask] = 0
            # tmp = F.softmax(tmp, dim=-1)
            # self.alphas= tmp #! this will cause optimizer to update old object
            # self.alphas = torch.nn.Parameter(tmp)
        # print("normalize alphas", self.alphas)
    def checkDropAlpha(self, epoch, kthSeed):
        #* to check whether drop alphas at particular epoch
        return  epoch in epoch_to_drop
    def forward(self, input):
        #* every time use alphas need to set alphas to 0 which has been drop
        self.filtAlphas()
        # print("foward()")
        output = self.layerDict["layer_0"](input, self.alphas[0])
        # print("output layer_0", output)
        output = self.layerDict["max_pool1"](output)
        output = self.layerDict["layer_1"](output, self.alphas[1])
        output = self.layerDict["layer_2"](output, self.alphas[2])
        # print("output layer_2", output)
        output = self.layerDict["max_pool2"](output)
        output = self.layerDict["layer_3"](output , self.alphas[3])
        output = self.layerDict["layer_4"](output , self.alphas[4])
        output = self.layerDict["max_pool3"](output)
        # output = self.layerDict["layer_5"](output , self.alphas[5])
        
        # print("self.layerDict[layer_5](output , self.alphas[5])", output)
        #! 先關閉layer3 layer4 增加訓練速度
        # output = self.layerDict["layer_6"](output , self.alphas[6])
        # print("self.layerDict[layer_6](output , self.alphas[6]", output)
        # output = self.layerDict["layer_7"](output , self.alphas[7])
        # print("self.layerDict[layer_67](output , self.alphas[7]", output)
        # output = self.layerDict["layer_8"](output , self.alphas[8])
        # print("self.layerDict[layer_8](output , self.alphas[8]", output)
        # output = self.layerDict["layer_9"](output , self.alphas[9])
        # output = self.layerDict["max_pool3"](output)
        # print("self.layerDict[max_pool3](output)", output)
        # print("tensor with shape{} is going to fc".format(output.shape))
        output = torch.flatten(output, start_dim=1)
        # print("tensor with shape{} is going to fc".format(output.shape))
        # print("alphas", self.alphas)
        # print("alphas mask", self.alphasMask)
        # print("model", self)
        # exit()
        output = self.fc(output)
        return output
    def getAlphas(self):
        # print("getAlphas()")
        if hasattr(self, "alphasParameters"):
            # print("hasttr")
            return self.alphasParameters
        #! why returning a set is correct
        self.alphasParameters = [
            v for k, v in self.named_parameters() if k=="alphas"
        ]
        #! I think it should return a list
        #! Module.register_parameter() takes iterable parameter, and set is a iterable
        # self.alphasParameters = [
        #     v for k, v in self.named_parameters() if k=="alphas"
        # ]
        print("\nalphas id in getAlphas()\n", id(self.alphas))
        return self.alphasParameters
    def getAlphasTensor(self):
        # print("getAlphas()")
        if hasattr(self, "alphasParameters"):
            return torch.Tensor(self.alphasParameters[0].cpu().clone().detach())
        #! why returning a set is correct
        self.alphasParameters = [
            v for k, v in self.named_parameters() if k=="alphas"
        ]
        return torch.Tensor(self.alphasParameters[0].clone().detach())
    def getWeight(self):
        if hasattr(self, "weightParameters"):
            return self.weightParameters

        self.weightParameters = {
            v for k, v in self.named_parameters() if k!="alphas"
        }

        return self.weightParameters




if __name__ =="__main__":
    layer = Model(5, 2, 10)
    # print(list(layer.named_parameters()))
    tmp = {
        para for name, para in layer.named_parameters() if name == "alphas" 
    }
    for name, para in layer.named_parameters():

        if name == "alphas":
            pass
            # print("===", name)
            # tmp = {}
    # print(hasattr(layer, "_alphas"))
    # print(tmp)
    # print(type(layer.parameters()))
    for i in layer.parameters():
        pass
        # print(i)
    # print(layer)
    print(epoch_to_drop)
    for i in range(40):
        
        print(i in epoch_to_drop, i)