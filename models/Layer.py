import torch.nn as nn
import torch 
from  .InnerCell import InnerCell

class Layer(nn.Module):
    def __init__(self, numOfInnerCell, layer, inputChannel=3, outputChannel=96, stride=1, padding=1, cellArchPerLayer=None, layerName="", InnerCellArch=[1, 1, 1, 1, 1]):

        super(Layer, self).__init__()
        #info set private attribute
        self.numOfInnerCell = numOfInnerCell
        self.layer = layer
        self.inputChannel = inputChannel
        self.outputChannel = outputChannel
        self.layerName = layerName
        self.InnerCellArch = InnerCellArch
        self.alphasDict = {}
        self.alphasList = []
        self.betaDict = {}
        self.betaList = []
        
        #info define layer structure
        # print("cellArchPerLayer", cellArchPerLayer)
        self.innerCellDict = nn.ModuleDict({
            'innerCell_0': InnerCell(inputChannel, outputChannel//self.numOfInnerCell, stride, cellArchPerLayer[0], innercellName=self.layerName+".innerCell_0", opIndex=InnerCellArch),
            # 'innerCell_'+str(layer)+'_1': cell(inputChannel, outputChannel//self.numOfInnerCell, stride),
        })
        #info create alphaList 
        for innerCellName in self.innerCellDict:
            tmp = self.innerCellDict[innerCellName].getAlpha()
            self.alphasDict[innerCellName] = [ tmp ]
            self.alphasList.append(tmp)
            
        #info create betaList
        for innerCellName in self.innerCellDict:
            tmp = self.innerCellDict[innerCellName].getBeta()
            self.betaDict[innerCellName] = [tmp]
            self.betaList.append(tmp)
    def turnSwitchAt(self, innerCellName, onOrOff):
        self.innerCellDict[innerCellName].turnInnerCellSwitch(onOrOff)
    def getSwitchAt(self, innerCellName):
        return self.innerCellDict[innerCellName].getInnerCellSwitch()
    def getBeta(self):
        return self.betaList
    def setBeta(self, innerCellName, inputBeta ):
        self.innerCellDict[innerCellName].setBeta(inputBeta)
    def remakeRemainOp(self):
        for name in self.innerCellDict:
            self.innerCellDict[name].remakeRemainOp()
    def forward(self, input):
        #* concate innerCell's output instead of add them elementwise
        
        output = None

        for name in self.innerCellDict:
            # add each inner cell directly without alphas involved
                # check whether inner cell still be used
            if output == None:
                output = self.innerCellDict[name](input)
            else:
                output = torch.cat( (output, self.innerCellDict[name](input)), dim=1 )
        if not hasattr(self, "emptyOutput"):
            self.emptyOutput = output*0
        return output
    
    def getAlphas(self):
        return self.alphasList
if __name__=="__main__":
    layer = Layer(1, 1)
    print(layer)