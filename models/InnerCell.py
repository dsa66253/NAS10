import torch
import torch.nn as nn
from data.config import PRIMITIVES
from .myoperation import OPS
class InnerCell(nn.Module):
    #todo make it general def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell, alphas)
    #todo due to adding opIndex parameter, mymodel.py needs to be adjust
    def __init__(self, inputChannel, outputChannel, stride, cellArchPerIneerCell, innercellName, opIndex=[]):
        super(InnerCell, self).__init__()
        self.transArchPerInnerCell= []
        self.cellArchPerIneerCell = cellArchPerIneerCell
        self.innercellName = innercellName
        self.beta =  nn.Parameter(torch.FloatTensor([1]))
        self.innerCellSwitch = True
        #info trainslate index to key of operations
        for index in range(len(opIndex)):
            if opIndex[index]==1:
                self.transArchPerInnerCell.append(PRIMITIVES[index])
        #info make operations to a list according cellArchPerIneerCell
        self.opDict = nn.ModuleDict()
        # self.remainOpDict = nn.ModuleDict()
        self.alphasList = []
        for opName in self.transArchPerInnerCell:
            op = OPS[opName](inputChannel, outputChannel, stride, False, False)
            self.opDict[opName] = op
            # self.remainOpDict[opName] = op
            self.alphasList.append(op.getAlpha())
    def turnInnerCellSwitch(self, onOrOff):
        if onOrOff==0 or onOrOff==False:
            self.innerCellSwitch = False
        else:
            self.innerCellSwitch = True
    def getInnerCellSwitch(self):
        return self.innerCellSwitch
    def getBeta(self):
        return self.beta
    def setBeta(self, inputBeta):
        with torch.no_grad():
            self.beta *= 0
            self.beta += inputBeta
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
    def remakeRemainOp(self):
        #info the function can actually be implemented in turnOffOp()
        deleteNameList = []
        for key in self.remainOpDict:
            if not self.remainOpDict[key].getSwitch():
                deleteNameList.append(key)
        for i in range(len(deleteNameList)):
            del self.remainOpDict[deleteNameList[i]]
    def forward(self, input):
        #info add each output of operation element-wise
        # print("next(model.parameters()).is_cuda", next(self.parameters()).is_cuda)
        # out = self.opList[0](input)
        
            
        output = None
        for opName in self.opDict:
            #! Can NOT use inplace operation +=. WHY? 
            #! Ans: inplace operation make computational graph fail
            if output==None:
                output = self.opDict[opName](input) * self.opDict[opName].getAlpha()
            else:
                output = output + self.opDict[opName](input)* self.opDict[opName].getAlpha()

        output = output*self.beta
        #todo need to clarify responsibily of who need to multiply beta/alpha
        #todo it seems like all responsiblity is on InnerCell
        return output
    
if __name__=="__main__":
    pass