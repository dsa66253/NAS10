import torch.nn as nn
import torch
import os
from models.myoperation import OPS
from data.config import cfg_newnasmodel as cfg
from data.config import PRIMITIVES, featureMap
import numpy as np
from models.initWeight import initialize_weights
import random

class InnerCell(nn.Module):
    #todo make it general def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell, alphas)
    def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell, innercellName):
        super(InnerCell, self).__init__()
        #info private data
        self.cellArchPerIneerCell = cellArchPerIneerCell
        self.innercellName = innercellName
        
        #info make operations to a list according cellArchPerIneerCell
        self.opDict = nn.ModuleDict()
        for opIndex in range(len(self.cellArchPerIneerCell)):
            if self.cellArchPerIneerCell[opIndex]==1:
                op = OPS[PRIMITIVES[opIndex]](inputChannel, outputChannel, stride, False, False)
                # op.setAlpha(1) #* don't need to train alpha
                self.opDict[PRIMITIVES[opIndex]] = op
                
    def forward(self, input):
        #info add each output of operation element-wise
        output = None
        for opName in self.opDict:
            #! Can NOT use inplace operation +=. WHY? 
            #! Ans: inplace operation make computational graphq fail
            if self.opDict[opName].getSwitch():
                if output==None:
                    output = self.opDict[opName](input) * self.opDict[opName].getAlpha()
                else:
                    output = output + self.opDict[opName](input)* self.opDict[opName].getAlpha()
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
        
        #info define layer structure
        # print("cellArchPerLayer", cellArchPerLayer)
        self.innerCellDict = nn.ModuleDict({
            'innerCell_0': InnerCell(inputChannel, outputChannel//self.numOfInnerCell, stride, cellArchPerLayer, innercellName=self.layerName+".innerCell_0"),
            # 'innerCell_'+str(layer)+'_1': cell(inputChannel, outputChannel//self.numOfInnerCell, stride),
        })
        

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
        
        #info define network structure
        self.layerDict = nn.ModuleDict({})
        for key in cellArch:
            [_, i, j] = key.split("_")
            if i=="0":
                self.layerDict["layer_{}_{}".format(i, j)] = Layer(self.numOfInnerCell, 0, featureMap["f"+i]["channel"], featureMap["f"+j]["channel"], 4, cellArchPerLayer=cellArch[key], layerName=key)
            else:
                self.layerDict["layer_{}_{}".format(i, j)] = Layer(self.numOfInnerCell, 0, featureMap["f"+i]["channel"], featureMap["f"+j]["channel"], 1, cellArchPerLayer=cellArch[key], layerName=key)
        
        self.poolDict = nn.ModuleDict({})
        for i in range(3):
            self.poolDict["maxPool_{}".format(i)] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fc = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, cfg["numOfClasses"])
        )
    def __simpleForward(self, input):
        #* every time use alphas need to set alphas to 0 which has been drop
        # self.normalizeAlphas()
        #* manually create forward order
        output = self.layerDict["layer_0_1"](input)
        output = self.poolDict["maxPool_0"](output)
        output = self.layerDict["layer_1_2"](output)
        output = self.poolDict["maxPool_1"](output)
        output = self.layerDict["layer_2_3"](output)
        output = self.layerDict["layer_3_4"](output)
        output = self.layerDict["layer_4_5"](output)
        output = self.poolDict["maxPool_2"](output)
        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)
        return output
    def forward(self, input):
        return self.__simpleForward(input)
        #* create outputList
        # print("input;shape", input.shape)
        outputList = [[0] * len(featureMap) for i in range(len(featureMap))]
        featureMapList = [0]*len(featureMap)
        featureMapList[0] = input

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
        for i in range(len(featureMap)-1):
            for j in range(i+1, len(featureMap)):
                #* feed Fi feature map to layer
                # print("featureMapList[i]", featureMapList[i].shape)
                try:
                    output = self.layerDict["layer_{}_{}".format(i, j)](featureMapList[i])
                except KeyError:
                    break
                except Exception as e:
                    print("not key error", e)
                #* handle max pooling
                output = self.maxPool(j, output)
                outputList[i][j] = output
                # print("layer_{}_{}".format(i, j), "output.shape", output.shape)
            #* prepare the next feature map
            preOutput = None
            for nextFeatureMap in range(i+1): 
                if preOutput==None:
                    preOutput = outputList[nextFeatureMap][i+1]

                else:
                    try:
                        preOutput = preOutput + outputList[nextFeatureMap][i+1]
                    except:
                        print("unmatch dim on ", nextFeatureMap)
                        print(preOutput.shape, outputList[nextFeatureMap][i+1].shape)
                        exit()
            
            featureMapList[i+1] = preOutput
        output = featureMapList[-1]
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