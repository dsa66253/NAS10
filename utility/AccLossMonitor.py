import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import numpy as np
class AccLossMonitor():
    def __init__(self, kth, plotFolder, npFolder, trainType=""):
        self.kth = kth
        self.plotFolder = plotFolder
        self.npFolder = npFolder
        self.trainType = trainType #* Nas or retrain
    def saveAccLossNp(self, accRecord, lossRecord):
        print("save record to ", self.npFolder)
        try:
            np.save(os.path.join(self.npFolder, "{}_train_loss_{}".format(self.trainType, self.kth)), lossRecord["train"])
            np.save(os.path.join(self.npFolder, "{}_val_loss_{}".format(self.trainType, self.kth)), lossRecord["val"])
            np.save(os.path.join(self.npFolder, "{}_train_acc_{}".format(self.trainType, self.kth)), accRecord["train"])
            np.save(os.path.join(self.npFolder, "{}_val_acc_{}".format(self.trainType, self.kth)), accRecord["val"])
            np.save(os.path.join(self.npFolder, "{}_test_acc_{}".format(self.trainType, self.kth)), accRecord["test"])
        except Exception as e:
            print("Fail to save acc and loss")
            print(e)
    def plotAccLineChart(self, plotDict, tag=""):
        fig, ax = plt.subplots()
        ax.plot(plotDict['train'], c='tab:red', label='train')
        ax.plot(plotDict['val'], c='tab:cyan', label='val')
        try:
            ax.plot(plotDict['test'], c='tab:brown', label='test')
        except Exception as e:
            print("null accRecord['test']", e)
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.title('acc of {}'.format(self.kth))
        
        plt.legend()
        filename = "acc_{}th".format(self.kth)
        plt.savefig(os.path.join(self.plotFolder, filename))
    def plotLossLineChart(self, plotDict, tag=""):
        fig, ax = plt.subplots()
        ax.plot(plotDict['train'], c='tab:red', label='train')
        ax.plot(plotDict['val'], c='tab:cyan', label='val')
        try:
            ax.plot(plotDict['test'], c='tab:brown', label='test')
        except Exception as e:
            print("null lossRecord['test']", e)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('loss of {}'.format(self.kth))
        plt.legend()
        filename = "loss_{}th".format(self.kth)
        plt.savefig(os.path.join(self.plotFolder, filename))