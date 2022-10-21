
import matplotlib.pyplot as plt
import numpy as np
from models.myoperation import Conv, OPS
import torch
import torch.nn as nn
from models.mymodel import Model
from models.arch import simpleArch
import os
torch.set_printoptions(precision=4, sci_mode=False)
class HistDrawer():
    def __init__(self, saveFolder) -> None:
        self.saveFolder = saveFolder
    def drawNetConvWeight(self, net, tag="ori"):
        for k, v in net.named_modules():
            if isinstance(v, nn.Conv2d):
                print(k)
                nameSplit = k.split(".")
                fileName = tag + "." + nameSplit[1] + "." + nameSplit[-3] 
                self.drawHist(v.weight, fileName, tag)
                self.drawHist(v.bias, fileName+"bias", tag)
    
    def drawHist(self, tensor, fileName, tag=""):
        # print(tensor.sum())
        fig, ax = plt.subplots(2, 1, figsize=(5, 2.7), layout='constrained')
        # print(torch.mean(tensor), torch.std(tensor), torch.max(tensor), torch.min(tensor))
        # ax.set_position([box.x0, 1, box.width * 0.8, box.height])
        table = [["mean", torch.mean(tensor).item()], 
        ["std", torch.std(tensor).item()],
        ["max", torch.max(tensor).item()],
        ["min", torch.min(tensor).item()]]
        
        background = [["w", "w"],
        ["w", "w"],
        ["w", "w"],
        ["w", "w"]]
        # print(tensor)
        table = ax[0].table(table, background, bbox=[0, 0, 1, 1])
        table.set_fontsize(50)
        binCount = np.arange(torch.min(tensor).item(), torch.max(tensor).item(), 0.005)
        ax[1].hist(np.reshape(tensor.data.cpu().numpy(), (-1)), bins=100)
        
        
        try:
            normalTensor = tensor.data.detach().clone().cpu()
            torch.nn.init.normal_(normalTensor, torch.mean(tensor).item(), torch.std(tensor).item())
            ax[1].hist(np.reshape(normalTensor.numpy(), (-1)), bins=100, alpha=0.3, color="orange")
        except Exception as e:
            print(e)
        # box = ax.get_position()
        # ax.set_position([box.x0, 1, box.width * 0.8, box.height])
        # # print(bin[0], len(bin[0]))
        # # print(bin[1], len(bin[1]))
        # # print(bin[2], len(bin[2]))
        print("save to ", os.path.join(self.saveFolder, fileName)+".png")
        
        plt.savefig(os.path.join(self.saveFolder, fileName)+".png")
        plt.close()

if __name__=="__main__":
    histDrawer = HistDrawer("./")
    op = OPS["conv_5x5"](96, 128, 1, 1, 1)
    # net = Model(simpleArch)
    # histDrawer.drawNetConvWeight(net)
    # for k, v in op.named_modules():
    #     if isinstance(v, nn.Conv2d):
    #         print(k, v)
    #         for key, value in v.named_parameters():
    #             histDrawer.drawHist(v, "")
    # for k, v in op.named_parameters():
    #     print(k)
        # if "op.0.weight"==k:
        #     print(v.shape)
        #     print(type(v))
        #     histDrawer.drawHist(v, "")

    # fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    # bin = ax.hist(np.reshape(netLyaerPara.data.cpu().numpy(), (-1)), bins=50)
    # # print(bin[0], len(bin[0]))
    # # print(bin[1], len(bin[1]))
    # # print(bin[2], len(bin[2]))
    # plt.savefig("./hist")