import torch.nn as nn
import torch
import os
from models.alpha.operation import OPS
from data.config import cfg_newnasmodel as cfg
from data.config import PRIMITIVES
import numpy as np
from utility.initWeight import initialize_weights
import random
class InnerCell(nn.Module):
    #todo make it general def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell, alphas)
    def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell):
        super(InnerCell, self).__init__()
        self.cellArchPerIneerCell = cellArchPerIneerCell
        #Todo OPS[primitive]
        #todo maybe make it a dictionary layer_i_j_convName
        #info make operations to a list according cellArchPerIneerCell

        self.opList = nn.ModuleList()
        for opName in self.cellArchPerIneerCell:
            self.opList.append(OPS[opName](inputChannel, outputChannel, stride, False, False))
        
    def forward(self, input):
        #info add each output of operation element-wise
        # print("next(model.parameters()).is_cuda", next(self.parameters()).is_cuda)
        numOfOp = len(self.opList)
        output = torch.zeros(self.opList[0](input).shape).to(input.device)
        countOp = 0
        for op in self.opList:
            output = output + op(input)
        # print("self.opList", self.opList)
        # print("output", output)
        #! it need to be fixed here
        # for op in self.opList:
        #     output = output + op(input)
            #! Can NOT use inplace operation +=. WHY?
        
        return output

class Layer(nn.Module):
    def __init__(self, numOfInnerCell, layer, cellArchPerLayer,  inputChannel=3, outputChannel=96, stride=1, padding=1):
        super(Layer, self).__init__()
        #info set private attribute
        self.name = "layer_"+str(layer)
        self.numOfInnerCell = numOfInnerCell
        self.layer = layer
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        self.cellArchPerLayer = cellArchPerLayer
        
        #todo according cellArch, dynamically create dictionary, and pass it to ModuleDict
        #info set inner cell

        self.innerCellDic = nn.ModuleDict({
            'innerCell_'+str(layer)+'_0': InnerCell(inputChannel, outputChannel//self.numOfInnerCell, stride, self.cellArchPerLayer[0]),
            # 'innerCell_'+str(layer)+'_1': InnerCell(inputChannel, outputChannel//self.numOfInnerCell, stride, self.cellArchPerLayer[1])
        })
    
    def forward(self, input):
        #* concate innerCell's output instead of add them elementwise
        indexOfInnerCell = 0
        output = 0
        for name in self.innerCellDic:
            # add each inner cell directly without alphas involved
            if indexOfInnerCell == 0:
                output = self.innerCellDic[name](input)
            else:
                output = torch.cat( (output, self.innerCellDic[name](input) ), dim=1 )

            indexOfInnerCell = indexOfInnerCell + 1
            # print("innerCellList{} output".format(name), output)
        return output
    
class NewNasModel(nn.Module):
    def __init__(self, cellArch):
        super(NewNasModel, self).__init__()
        #info private attribute
        self.numOfClasses = cfg["numOfClasses"]
        self.numOfOpPerCell = cfg["numOfOperations"]
        self.numOfLayer = cfg["numOfLayers"]
        self.numOfInnerCell = cfg["numOfInnerCell"]
        self.cellArch = cellArch
        print("cellArch", cellArch)
        self.cellArchTrans = self.translateCellArch()

        # print("self.cellArchTrans", self.cellArchTrans)
        #info network structure
        
        self.layerDict = nn.ModuleDict({
            "layer_0":Layer(self.numOfInnerCell, 0, self.cellArchTrans[0], 3, 96, 4),
            "layer_1":Layer(self.numOfInnerCell, 1, self.cellArchTrans[1], 96, 256, 1),
            "layer_2":Layer(self.numOfInnerCell, 2, self.cellArchTrans[2], 256, 384, 1),
            "layer_3":Layer(self.numOfInnerCell, 3, self.cellArchTrans[3], 384, 384, 1),
            "layer_4":Layer(self.numOfInnerCell, 4, self.cellArchTrans[4], 384, 256, 1),
            # "max_pool1":nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # "max_pool2": nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 'max_pool3': nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            "max_pool1":nn.MaxPool2d(kernel_size=3, stride=2),
            "max_pool2": nn.MaxPool2d(kernel_size=3, stride=2),
            'max_pool3': nn.MaxPool2d(kernel_size=3, stride=2)
        })
        
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.numOfClasses)
        )
        self._initialize_weights()
    def forward(self, input):
        # print("next(model.parameters()).is_cuda", next(self.parameters()).is_cuda)
        # print(input.shape)

        # set_seed_cpu(20)
        output = self.layerDict["layer_0"](input)
        # print("layer_0", output.shape)
        output = self.layerDict["max_pool1"](output)
        # print("max_pool1", output.shape)
        output = self.layerDict["layer_1"](output)
        # print("layer_1", output.shape)
        output = self.layerDict["max_pool2"](output)
        # print("max_pool2", output.shape)
        output = self.layerDict["layer_2"](output)
        # print("layer_2", output.shape)
        output = self.layerDict["layer_3"](output)
        # print("layer_3", output.shape)
        output = self.layerDict["layer_4"](output)
        # print("layer_4", output.shape)
        output = self.layerDict["max_pool3"](output)
        
        # print("tensor with shape{} is going to fc".format(output.shape))
        
        output = torch.flatten(output, start_dim=1)
        #todo keep batch size and match size of output to input of fc
        # print("output", output)
        output = self.fc(output)
        
        return output
    def translateCellArch(self):
        #* transform index of architecture to 2D list of PRIMITIVES string in ./config.py
        #! cannot get other package data
        cellArchTrans = []
        cellArchFlat = self.cellArch.flatten()
        tmp = []
        for opIndex in cellArchFlat:
            cellArchTrans.append(PRIMITIVES[opIndex])
        cellArchTrans = np.reshape(cellArchTrans, self.cellArch.shape)
        return cellArchTrans
        for i in range(self.cellArch.shape[0]):
            tmp = []
            for j in range(self.cellArch.shape[1]):
                tmp.append(PRIMITIVES[self.cellArch[i][j]])
            cellArchTrans.append(tmp)
        return cellArchTrans
        # print(string)
    def _initialize_weights(self):
        initialize_weights(self)
        # info make all weight positive
        # for netLayerName , netLayerPara in self.named_parameters():

        #     tmp = torch.abs(netLayerPara)
            
        #     # netLayerPara = torch.mul(netLayerPara, 0)
        #     netLayerPara = netLayerPara*0
        #     print("netLayerPara", netLayerPara)
            # netLayerPara+=tmp
        
        
        
def set_seed_cpu(seed):
    # print("set_seed_cpu seed: ", seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
if __name__ == '__main__':
    genotype_filename = os.path.join('./weights_pdarts_nodrop/',
                        'genotype_' + str(0) +".npy")
    # np.load(genotype_filename)
    
    # print(OPS)
    cellArch = np.load(genotype_filename)
    print(cellArch)
    model = NewNasModel(1, 2, 3, cellArch)
    model.translateCellArch()
    exit()
    arr = np.random.rand(3,2)
    
    string = np.empty(arr.shape, dtype=np.unicode_)
    print(arr)
    string = []
    print(arr.shape[0])
    print(arr.shape[1])
    for i in range(arr.shape[0]):
        tmp = []
        for j in range(arr.shape[1]):
            
            tmp.append(str(arr[i][j]))
        string.append(tmp)
    print(string)
    # model.translateCellArch