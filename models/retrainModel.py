import torch.nn as nn
import torch
import os
from models.myoperation import OPS, Conv
from data.config import cfg_newnasmodel as cfg
from data.config import PRIMITIVES, featureMap
import numpy as np
from models.initWeight import initialize_weights
import random
from .Layer import Layer
from .InnerCell import InnerCell

class NewNasModel(nn.Module):
    def __init__(self, cellArch):
        super(NewNasModel, self).__init__()
        #info private attribute
        self.numOfClasses = cfg["numOfClasses"]
        self.numOfOpPerCell = cfg["numOfOperations"]
        self.numOfLayer = cfg["numOfLayers"]
        self.numOfInnerCell = cfg["numOfInnerCell"]
        self.cellArch = cellArch
        self.alphasDict = {}
        # self.cellArchTrans = self.translateCellArch()
        # print("self.cellArch", self.cellArch)
        #info define network structure
        self.layerDict = nn.ModuleDict({})
        for key in cellArch:
            [_, i, j] = key.split("_")
            if i=="0":
                if sum(cellArch[key])>0:
                    self.layerDict["layer_{}_{}".format(i, j)] = Layer(self.numOfInnerCell, 0, featureMap["f"+i]["channel"], featureMap["f"+j]["channel"], 4, cellArchPerLayer=cellArch[key], layerName=key, InnerCellArch=cellArch[key])
            else:
                if sum(cellArch[key])>0:
                    self.layerDict["layer_{}_{}".format(i, j)] = Layer(self.numOfInnerCell, 0, featureMap["f"+i]["channel"], featureMap["f"+j]["channel"], 1, cellArchPerLayer=cellArch[key], layerName=key, InnerCellArch=cellArch[key])
        
        self.poolDict = nn.ModuleDict({})
        for i in range(3):
            self.poolDict["maxPool_{}".format(i)] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, cfg["numOfClasses"])
        )
        # info set initial beta and alpha
        for k, v in self.named_modules():
            if isinstance(v, InnerCell):
                v.setBeta(1.0)
            if isinstance(v, Conv):
                v.setAlpha(1.0)
        
        
    def forward(self, input):
        output=None
        for layerName in self.layerDict:
            if output==None:
                output = self.layerDict[layerName](input)
            else:
                output = self.layerDict[layerName](output)
            output = self.maxPool(layerName[-1], output)
        
        # #* create outputList
        # # print("input;shape", input.shape)
        # outputList = [[0] * len(featureMap) for i in range(len(featureMap))]
        # featureMapList = [0]*len(featureMap)
        # featureMapList[0] = input

        # for i in range(len(featureMap)-1):
        #     for j in range(i+1, len(featureMap)):
        #         #* feed Fi feature map to layer
        #         # print("featureMapList[i]", featureMapList[i].shape)
        #         output = self.layerDict["layer_{}_{}".format(i, j)](featureMapList[i])

        #         #* handle max pooling
        #         output = self.maxPool(j, output)
        #         outputList[i][j] = output
        #         # print("layer_{}_{}".format(i, j), "output.shape", output.shape)
        #     #* prepare the next feature map
        #     preOutput = None
        #     for nextFeatureMap in range(i+1): 
        #         if preOutput==None:
        #             preOutput = outputList[nextFeatureMap][i+1]
        #         else:
        #             try:
        #                 preOutput = preOutput + outputList[nextFeatureMap][i+1]
        #             except:
        #                 print("unmatch dim on ", nextFeatureMap)
        #                 print(preOutput.shape, outputList[nextFeatureMap][i+1].shape)
        #                 exit()

        #     featureMapList[i+1] = preOutput
        # output = featureMapList[-1]
        #info to fc layer
        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)
        return output
    def maxPool(self, targetFeatureMap, input):
        key = "f"+str(targetFeatureMap)
        output = input
        # print("current dim", output.shape[2], " target dim ", featureMap[key]["featureMapDim"])
        for i in range(len(featureMap)):
            if output.shape[2]>featureMap[key]["featureMapDim"]:
                #* to see if it reduce to next feature map dimension 
                output = self.poolDict["maxPool_0"](output)
                # print("current dim", output.shape[2], " target dim ", featureMap[key]["featureMapDim"])
            else:
                break
        return output
    def __initialize_weights(self):
        initialize_weights(self)
    def getWeight(self):
        print("getWeight()")
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
                    for key, para in v.named_parameters():
                        self.weightParameters.append(para)
            elif "poolDict" in k.split(".")[-1] or "fc" in k.split(".")[-1]:
                print(k)
                for key, para in v.named_parameters():
                    self.weightParameters.append(para)
        return self.weightParameters
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