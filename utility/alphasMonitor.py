import torch
from data.config import folder 
import numpy as np
import os
import numpy as np
class AlphasMonitor():
    def __init__(self):
        self.allAlphas = 0
        self.allAlphasGrad = 0
        self.alphaLogDict = None
    def logAlphaDictPerIter(self, net, iteration):
        alphaDict = net.getAlphaDict()
        if self.alphaLogDict==None:
            self.alphaLogDict = {}
            for layerName in alphaDict:
                self.alphaLogDict[layerName] = torch.FloatTensor(alphaDict[layerName]).numpy()
        else:
            for layerName in alphaDict:
                self.alphaLogDict[layerName] = np.append(self.alphaLogDict[layerName], torch.FloatTensor(alphaDict[layerName]).numpy(), axis=0)


    def logAlphasPerIteration(self, net, iteration):
        if iteration==0:
            tmp = net.getAlphasTensor()
            self.allAlphas = tmp.reshape((1, *tmp.size()))
        else:
            tmp = net.getAlphasTensor()
            self.allAlphas = torch.cat((self.allAlphas, tmp.reshape((1, *tmp.size()))))
    def saveAllAlphas(self, kth):
        for layerName in self.alphaLogDict:
            try:
                np.save(os.path.join( folder["alpha_pdart_nodrop"], "{}th_{}".format(kth, layerName) ), self.alphaLogDict[layerName])
            except Exception as e:
                print("cannot save alphasPerIteration", e)
            
    def logAlphasGradPerIteration(self, net, iteration):
        if iteration==0:
            
            tmp = net.alphas.grad
            print("tmp = net.alphas.grad", net.alphas.grad)
            self.allAlphasGrad = tmp.reshape((1, *tmp.size()))
        else:
            tmp = net.alphas.grad
            self.allAlphasGrad = torch.cat((self.allAlphasGrad, tmp.reshape((1, *tmp.size()))))
    def saveAllAlphasGrad(self, kth):
        try:
            np.save(os.path.join( folder["alpha_pdart_nodrop"], "allAlphasGrad_{}".format(kth) ), self.allAlphasGrad.cpu().detach().numpy())
        except Exception as e:
            print("cannot save alphasGradPerIteration", e)
            