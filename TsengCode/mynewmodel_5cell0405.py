
import torch.nn as nn

import torch
import os
from models.alpha.operation import OPS
# from data.config import PRIMITIVES
import numpy as np


class Cell_conv(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        # self.mix_op = MixedOp_conv(c_in, c_out, stride)

    def forward(self, x, weights):
        y = self.mix_op(x, weights)  # x is input , weights is alpha value
        return y


class InnerCell(nn.Module):
    #todo make it general def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell, alphas)
    def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell):
        super(InnerCell, self).__init__()
        self.cellArchPerIneerCell = cellArchPerIneerCell
        #Todo OPS[primitive]
        #todo maybe make it a dictionary layer_i_j_convName
        #info make operations to a list according cellArchPerIneerCell
        self.opList = nn.ModuleList()
        for convName in self.cellArchPerIneerCell:
            self.opList.append(OPS[convName](inputChannel, outputChannel, stride, False, False))
        
    def forward(self, input):
        #info add each output of operation element-wise
        # print("next(model.parameters()).is_cuda", next(self.parameters()).is_cuda)
        out = self.opList[0](input)
        
        for op in self.opList:
            out = out + op(input)
            #! Can NOT use inplace operation +=. WHY?
        return out

class Layer(nn.Module):
    def __init__(self, numOfInnerCell, layer, cellArchPerLayer,  inputChannel=3, outputChannel=96, stride=1, padding=1):
        super(Layer, self).__init__()
        #info set private attribute
        self.name = "layer_"+str(layer)
        self.cellArchPerLayer = cellArchPerLayer
        #todo according cellArch, dynamically create dictionary, and pass it to ModuleDict
        #info set inner cell
        self.innerCellDic = nn.ModuleDict({
            'innerCell_'+str(layer)+'_0': InnerCell(inputChannel, outputChannel, stride, self.cellArchPerLayer[0]),
            'innerCell_'+str(layer)+'_1': InnerCell(inputChannel, outputChannel, stride, self.cellArchPerLayer[1]),
        })
    
    def forward(self, input):
        #* add each inner cell directly as layer's output without alphas involved
        indexOfInnerCell = 0
        output = 0
        
        for name in self.innerCellDic:
            # add each inner cell directly as layer's output without alphas involved
            output = output + self.innerCellDic[name](input)
            indexOfInnerCell = indexOfInnerCell + 1
        return output
    
class NewNasModel(nn.Module):
    def __init__(self,numOfLayer, numOfInnerCell, numOfClasses, cellArch):
        super(NewNasModel, self).__init__()
        #info private attribute
        self.numOfClasses = numOfClasses
        self.numOfOpPerCell = 5
        self.numOfLayer = numOfLayer
        self.numOfInnerCell = numOfInnerCell
        self.cellArch = cellArch
        
        self.cellArchTrans = self.translateCellArch()
        
        self.layerDict = nn.ModuleDict({
            "layer_0":Layer(self.numOfInnerCell, 0, self.cellArchTrans, 3, 96, 4),
            "layer_1":Layer(self.numOfInnerCell, 1, self.cellArchTrans, 96, 256, 1),
            "layer_2":Layer(self.numOfInnerCell, 2, self.cellArchTrans, 256, 384, 1),
            "layer_3":Layer(self.numOfInnerCell, 3, self.cellArchTrans, 384, 384, 1),
            "layer_4":Layer(self.numOfInnerCell, 4, self.cellArchTrans, 384, 256, 1),
            "max_pool1":nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            "max_pool2": nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            'max_pool3': nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        })
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.numOfClasses)
        )
    def forward(self, input):
        # print("next(model.parameters()).is_cuda", next(self.parameters()).is_cuda)
        # print(input.shape)
        output = self.layerDict["layer_0"](input)
        output = self.layerDict["max_pool1"](output)
        output = self.layerDict["layer_1"](output)
        output = self.layerDict["max_pool2"](output)
        output = self.layerDict["layer_2"](output)
        output = self.layerDict["layer_3"](output)
        output = self.layerDict["layer_4"](output)
        output = self.layerDict["max_pool3"](output)
        output = torch.flatten(output, start_dim=1)
        #todo keep batch size and match size of output to input of fc
        output = self.fc(output)
        return output
    def translateCellArch(self):
        #* transform index of architecture to 2D list of PRIMITIVES string in ./config.py
        #! cannot get other package data
        cellArchTrans = []
        PRIMITIVES = [
            'conv_3x3',
            'conv_5x5',
            'conv_7x7',
            'conv_9x9',
            'conv_11x11',
        ]
        for i in range(self.cellArch.shape[0]):
            tmp = []
            for j in range(self.cellArch.shape[1]):
                tmp.append(PRIMITIVES[self.cellArch[i][j]])
            cellArchTrans.append(tmp)
        return cellArchTrans
        # print(string)
    
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