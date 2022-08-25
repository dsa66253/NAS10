from .myoperation import OPS, Conv
import torch
import torch.nn as nn
import torch.optim as optim
from data.config import PRIMITIVES, trainMatrix, featureMapDim, featureMap
from data.config import cfg_nasmodel as cfg
import os
import torch.nn.functional as F
import numpy as np
from data.config import epoch_to_drop


#info Cell_conv actually is innerCell
class InnerCell(nn.Module):
    #todo make it general def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell, alphas)
    def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell, innercellName):
        super(InnerCell, self).__init__()
        # trainslate index to key of operations
        self.transArchPerInnerCell= []
        for index in range(len(PRIMITIVES)):
            self.transArchPerInnerCell.append(PRIMITIVES[index])
        self.cellArchPerIneerCell = cellArchPerIneerCell
        self.innercellName = innercellName

        #info make operations to a list according cellArchPerIneerCell
        self.opDict = nn.ModuleDict()
        self.alphasList = []
        for opName in self.transArchPerInnerCell:
            op = OPS[opName](inputChannel, outputChannel, stride, False, False)
            self.opDict[opName] = op
            self.alphasList.append(op.getAlpha())
    def getOpDict(self):
        return self.opDict
    def getAlpha(self):
        # for i in range(len(self.opList)):
        #     print(id(self.alphasList[i]))
        # for i in range(len(self.opList)):
        #     self.alphasList.append(self.opList[i].getAlpha())
        #     print(id(self.alphasList[i]))
        #! this will get old alpha
        # exit()
        self.alphasList = []
        for opName in self.opDict:
            self.alphasList.append(self.opDict[opName].getAlpha())
        return self.alphasList
    def getAlphaAt(self, index):
        return self.opDict[PRIMITIVES[index]].getAlpha()
    def setAlpha(self, inputAlphaList):
        for i in range(len(self.opDict)):
            self.opDict[PRIMITIVES[i]].setAlpha(inputAlphaList[i])
    def setAlphaAt(self, index, value):
        self.opDict[PRIMITIVES[index]].setAlpha(value)
    def getSwitch(self):
        switchList = []
        for i in range(len(self.opDict)):
            switchList.append(self.opDict[PRIMITIVES[i]].getSwitch())
        return switchList
    def getSwitchAt(self, index):
        return self.opDict[PRIMITIVES[index]].getSwitch()
    def turnOffOp(self, index):
        self.opDict[PRIMITIVES[index]].turnSwitch(False)
    def show(self, input):
        output = None
        for opName in self.opDict:
            #! Can NOT use inplace operation +=. WHY? 
            #! Ans: inplace operation make computational graphq fail
            if self.opDict[opName].getSwitch():
                
                if output==None:
                    print("forward ", self.innercellName, opName, self.opDict[opName].getAlpha().is_leaf, self.opDict[opName].getAlpha().grad)
                    output = self.opDict[opName](input)
                else:
                    print("forward ", self.innercellName, opName, self.opDict[opName].getAlpha().is_leaf, self.opDict[opName].getAlpha().grad)
                    output = output + self.opDict[opName](input)
            
        return output
    def forward(self, input):
        #info add each output of operation element-wise
        # print("next(model.parameters()).is_cuda", next(self.parameters()).is_cuda)
        # out = self.opList[0](input)
        output = None
        for opName in self.opDict:
            #! Can NOT use inplace operation +=. WHY? 
            #! Ans: inplace operation make computational graphq fail
            if self.opDict[opName].getSwitch():
                
                if output==None:
                    # print("forward ", self.innercellName, opName, self.opDict[opName].getAlpha().is_leaf, self.opDict[opName].getAlpha().grad)
                    output = self.opDict[opName](input)
                else:
                    # print("forward ", self.innercellName, opName, self.opDict[opName].getAlpha().is_leaf, self.opDict[opName].getAlpha().grad)
                    output = output + self.opDict[opName](input)
            
        return output
class Layer(nn.Module):
    def __init__(self, numOfInnerCell, layer, inputChannel=3, outputChannel=96, stride=1, padding=1, cellArchPerLayer=None, layerName=""):
        super(Layer, self).__init__()
        #info set private attribute
        self.numOfInnerCell = numOfInnerCell
        self.layer = layer
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        self.layerName = layerName
        self.alphasDict = {}
        self.alphasList = []
        #info set inner cell
        
        # print("cellArchPerLayer", cellArchPerLayer)
        self.innerCellDict = nn.ModuleDict({
            'innerCell_0': InnerCell(inputChannel, outputChannel//self.numOfInnerCell, stride, cellArchPerLayer[0], innercellName=self.layerName+".innerCell_0"),
            # 'innerCell_'+str(layer)+'_1': cell(inputChannel, outputChannel//self.numOfInnerCell, stride),
        })
        
        for innerCellName in self.innerCellDict:
            tmp = self.innerCellDict[innerCellName].getAlpha()
            self.alphasDict[innerCellName] = [ tmp ]
            self.alphasList.append(tmp)

    def forward(self, input):
        #* concate innerCell's output instead of add them elementwise
        output = None
        for name in self.innerCellDict:
            # add each inner cell directly without alphas involved
            if output == None:
                output = self.innerCellDict[name](input)
            else:
                output = torch.cat( (output, self.innerCellDict[name](input)), dim=1 )

        return output
    def getAlphas(self):
        return self.alphasList

                    
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
        self.alphasDict = {}
        #info network structure
        self.layerDict = nn.ModuleDict({})
        for i in range(len(featureMapDim)-1):
            for j in range(i+1, len(featureMapDim)):
                if i==0:
                    self.layerDict["layer_{}_{}".format(i, j)] = Layer(self.numOfInnerCell, 0, featureMapDim[i], featureMapDim[j], 4, cellArchPerLayer=trainMatrix[0], layerName="layer_{}_{}".format(i, j))
                else:
                    self.layerDict["layer_{}_{}".format(i, j)] = Layer(self.numOfInnerCell, 0, featureMapDim[i], featureMapDim[j], 1, cellArchPerLayer=trainMatrix[0], layerName="layer_{}_{}".format(i, j))
                # print("layer_{}_{}".format(i, j), type(layerDic["layer_{}_{}".format(i, j)]))
        
        self.poolDict = nn.ModuleDict({})
        for i in range(3):
            self.poolDict["maxPool_{}".format(i)] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, cfg["numOfClasses"])
        )
        
        #info create alphaDict
        for k in self.layerDict.keys():
            if "layer" in k:
                self.alphasDict[k] = self.layerDict[k].getAlphas()

        self.initailizeAlphas()
        # self.alphasMask = torch.full(self._alphas.shape, False, dtype=torch.bool) #* True means that element are masked(dropped)
        
        #* self.alphas can get the attribute
    def getAlphasDict(self):
        return self.alphasDict

    def initailizeAlphas(self):
        #* set the first innercell's alpha to being evenly distributed, set second innercell's alphas to random
        for layerName in self.layerDict:
            for innerCellDict in self.layerDict[layerName].children():
                tmp = F.softmax(torch.FloatTensor(innerCellDict["innerCell_0"].getAlpha()) , dim=-1)
                innerCellDict["innerCell_0"].setAlpha(tmp)
        
    def dropMinAlpha(self):
        for layerName in self.layerDict:
            for innerCellDict in self.layerDict[layerName].children():
                #* for each innercell
                currentAlpha = torch.FloatTensor(innerCellDict["innerCell_0"].getAlpha())
                (_, allMinIndex) = torch.topk( currentAlpha, len(PRIMITIVES), largest=False )
                print("allMinIndex", allMinIndex, _)                
                #* drop operation
                for alphaIndex in allMinIndex:
                    if innerCellDict["innerCell_0"].getSwitchAt(alphaIndex)==True:
                        #* skip already drop operation
                        innerCellDict["innerCell_0"].turnOffOp(alphaIndex)
                        innerCellDict["innerCell_0"].setAlphaAt(alphaIndex, 0)
                        print("drop", layerName, PRIMITIVES[alphaIndex])
                        break
                print("after drop ", torch.FloatTensor(innerCellDict["innerCell_0"].getAlpha()))
                print("grad ", torch.FloatTensor(innerCellDict["innerCell_0"].getAlpha()).grad)
                
    
    
    def normalizeAlphas(self):
        #* set alphas to zero whose mask is true and pass it through softmax
        for layerName in self.layerDict:
            for innerCellDict in self.layerDict[layerName].children():
                #* for each innercell
                switchList = innerCellDict["innerCell_0"].getSwitch()
                alphaToNormalize = []
                alphaIndexToNormalize = []
                for opIndex in range(len(switchList)):
                    #* choose switch==True to normalize
                    if switchList[opIndex]:
                        alphaToNormalize.append(innerCellDict["innerCell_0"].getAlphaAt(opIndex))
                        alphaIndexToNormalize.append(opIndex)
                #* use softmax as normalize function
                alphaToNormalize = F.softmax(torch.FloatTensor(alphaToNormalize), dim=-1)
                for index in range(len(alphaIndexToNormalize)):
                    innerCellDict["innerCell_0"].setAlphaAt(alphaIndexToNormalize[index], alphaToNormalize[index])

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

    def checkDropAlpha(self, epoch, kthSeed):
        #* to check whether drop alphas at particular epoch
        return  epoch in epoch_to_drop
    def maxPool(self, targetFeatureMap, input):
        key = "f"+str(targetFeatureMap)
        output = input
        # print("current dim", output.shape[2], " target dim ", featureMap[key]["featureMapDim"])
        for i in range(len(featureMap)):
            if output.shape[2]>featureMap[key]["featureMapDim"]:
                output = self.poolDict["maxPool_0"](output)
                # print(output.shape, i)
            else:
                break
        return output

    def forward(self, input):
        #* every time use alphas need to set alphas to 0 which has been drop
        # self.normalizeAlphas()
        #* create outputList
        
        outputList = [[0] * len(featureMapDim) for i in range(len(featureMapDim))]
        featureMapList = [0]*len(featureMapDim)
        featureMapList[0] = input
        # print(outputList)
        # for ii in range(len(featureMapDim)):
        #     for jj in range(len(featureMapDim)):
        #         # print("type(outputList[ii][jj]", type(outputList[ii][jj]))
        #         if isinstance(type(outputList[ii][jj]), int):
        #             print(type(outputList[ii][jj]), end = ' ')
        #         else:
        #             print(type(outputList[ii][jj]), end = ' ')
        #     print()
        for i in range(len(featureMapDim)-1):
            for j in range(i+1, len(featureMapDim)):
                #* feed Fi feature map to layer
                output = self.layerDict["layer_{}_{}".format(i, j)](featureMapList[i])

                #* handle max pooling
                output = self.maxPool(j, output)
                outputList[i][j] = output
                # print("i ", i, " j ",j , "output.shape", output.shape)

                # for ii in range(len(featureMapDim)):
                #     for jj in range(len(featureMapDim)):
                #         print(type(outputList[ii][jj]), end = ' ')
                #     print()
            #* prepare the next feature map
            preOutput = None
            for nextFeatureMap in range(i+1): 
                if preOutput==None:
                    # print("preOutput==None")
                    # print("shape", outputList[nextFeatureMap][i+1].shape)
                    preOutput = outputList[nextFeatureMap][i+1]
                    # print("nextFeatureMap", nextFeatureMap, i+1, preOutput.shape)
                else:
                    # print("preOutput!=None")
                    # print("shape", outputList[nextFeatureMap][i+1].shape)
                    try:
                        # print("nextFeatureMap", nextFeatureMap, i+1, preOutput.shape)
                        preOutput = preOutput + outputList[nextFeatureMap][i+1]
                    except:
                        print(preOutput.shape, outputList[nextFeatureMap][i+1].shape)
                        exit()
                    
            # print("preOutput", i+1, preOutput.shape)
            

            # print("preOutput.shape", preOutput.shape)
            featureMapList[i+1] = preOutput
            # featureMapList.append(preOutput)
            # print("len(featureMapList)", len(featureMapList))
        output = featureMapList[-1]

        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)
        return output
    def getAlphaDict(self):
        return self.alphasDict
        
    def getAlphasPara(self):
        print("getAlphas()")
        #! why returning a set is correct
        self.alphasParameters = []
        for k, v in self.named_modules():
        #* algo: get all submodule, check if it's instance of Conv, check switch, renew optim
            if isinstance(v, Conv) and v.getSwitch()==True:
                # print("->",k , v)
                for key, para in v.named_parameters():
                    if "alpha" in key:
                        print(k)
                        self.alphasParameters.append(para)
                        print("grad ", para.grad)
        print(self.alphasParameters)
        return self.alphasParameters
    def getAlphasTensor(self):
        #! why returning a set is correct
        self.alphasParameters = [
            v for k, v in self.named_parameters() if k=="alphas"
        ]
        return torch.Tensor(self.alphasParameters[0].clone().detach())
    def getWeight(self):
        print("getWeight")
        # for k, v in self.named_parameters():
        #     print(k)
        # exit()
        # print(self.get_submodule("layerDict.layer_0_1.innerCellDict.innerCell_0.opList.0"))
        self.weightParameters = []
        for k, v in self.named_modules():
        #* algo: get all submodule, check if it's instance of Conv, check switch, renew optim
            # print("->", k)
            if "conv" in k.split(".")[-1]:
                # get conv module
                if v.getSwitch()==True:
                    print(k)
            elif "poolDict" in k.split(".")[-1] or "fc" in k.split(".")[-1]:
                print(k)
                for key, para in v.named_parameters():
                    self.weightParameters.append(para)
        return self.weightParameters
if __name__=="__main__":
    torch.manual_seed(10)
    innercell = InnerCell(3, 96, 1, None, "testLayer")
    print(innercell)
    exit()








    cretira = nn.CrossEntropyLoss()
    op = OPS["conv_3x3"](3, 2, 1, 1, 1)
    optimizer = optim.SGD(op.parameters(), lr=0.01, momentum=0.05)
    # print(op)
    op.show()
    
    sample = torch.rand((1, 3, 2, 2))
    label = torch.FloatTensor([[1.0, 0]])
    for i in range(10):
    # print("=============")
        y = op(sample)
        print(y.shape, label.shape)
        loss = cretira(y, label)
        loss.backward()
        optimizer.step()
        op.show()

    op.zero()
    op.show()
    print("=============")
    for i in range(1):
        # 
        y = op(sample)
        print(y.shape, label.shape)
        loss = cretira(y, label)
        loss.backward()
        optimizer.step()
        op.show()
    # print(op._alpha)