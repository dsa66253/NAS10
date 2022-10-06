import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os

def plot_loss_curve(lossRecord, title='default', saveFolder="./"):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    figure(figsize=(6, 4))
    plt.plot(lossRecord['train'], c='tab:red', label='train')
    plt.plot(lossRecord['val'], c='tab:cyan', label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss of {}'.format(title))
    plt.legend()
    
    plt.savefig(os.path.join(saveFolder, title))

def plot_acc_curve(accRecord, title='default', saveFolder="./"):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    fig, ax = plt.subplot()
    ax.plot(accRecord['train'], c='tab:red', label='train')
    ax.plot(accRecord['val'], c='tab:cyan', label='val')
    try:
        ax.plot(accRecord['test'], c='tab:brown', label='test')
    except Exception as e:
        print("null accRecord['test']", e)
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    ax.set_title(format(title))
    ax.legend()
    # plt.show()
    plt.savefig(os.path.join(saveFolder, title)) 
def plot_acc_curves(accRecord, ax, title='default', saveFolder="./"):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    totalEpoch = len(accRecord["train"])
    ax.plot(accRecord['train'], c='tab:red', label='train')
    ax.plot(accRecord['val'], c='tab:cyan', label='val')
    
    try:
        ax.plot(accRecord['test'], c='tab:brown', label='test')
    except Exception as e:
        print("null accRecord['test']", e)
    ax.yaxis.grid()
    ax.xaxis.grid()
    ax.set_yticks(range(0, 110, 10))
    ax.set_xticks(range(0, totalEpoch, 10))
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    ax.set_title(format(title))
    ax.legend()
def plot_combined_acc(folder = "./accLoss", title='combine', saveFolder="./plot", trainType="Nas"):
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for kth in range(3):
        trainNasTrainAccFile = os.path.join(folder, "{}_train_acc_{}.npy".format(trainType, str(kth)) )
        trainNasnValAccFile = os.path.join( folder,"{}_val_acc_{}.npy".format(trainType, str(kth)) )
        testAccFile = os.path.join( folder,"{}_test_acc_{}.npy".format(trainType, str(kth)) )
        # testAccFile = os.path.join(folder, "trainNasTestAcc_{}.npy".format(trainType, str(kth)) )
        try:
            accRecord = {
                "train": np.load(trainNasTrainAccFile),
                "val": np.load(trainNasnValAccFile),
                "test": np.load(testAccFile)
            }
        except:
            accRecord = {
                "train": np.load(trainNasTrainAccFile),
                "val": np.load(trainNasnValAccFile),
                # "test": np.load(testAccFile)
            }
        plot_acc_curves(accRecord, axs[kth], "acc_"+str(kth), "./plot")
    fileName = trainType+"_"+title
    print("save png to ", os.path.join(saveFolder, fileName))
    plt.savefig(os.path.join(saveFolder, fileName))


if __name__=="__main__":
    # plot_combined_acc(trainType="Nas")
    plot_combined_acc(trainType="retrain")
    # net = "alexnet"
    # folder = "./accLoss" 
    # title='combine_'+net
    # saveFolder="./plot"
    # fig, axs = plt.subplots(1, figsize=(10, 8), sharex=True, constrained_layout=True)
    # for kth in range(1):
    #     trainNasTrainAccFile = os.path.join(folder, "trainNasTrainAcc_{}.npy".format(str(kth)) )
    #     trainNasnValAccFile = os.path.join( folder,"trainNasValAcc_{}.npy".format(str(kth)) )
    #     testAccFile = os.path.join(folder, "trainNasTestAcc_{}.npy".format(str(kth)) )
        
        
    #     accRecord = {
    #         "train": np.load(trainNasTrainAccFile)*100,
    #         "val": np.load(trainNasnValAccFile)*100,
    #         "test": np.load(testAccFile)*100
    #         }
    #     plot_acc_curves(accRecord, axs, "acc_"+str(kth), "./plot")
    # # plt.show()
    # print("save png to ", os.path.join(saveFolder, title))
    # plt.savefig(os.path.join(saveFolder, title))
    exit()
    folder = "./accLoss"
    for kth in range(3):
        trainNasTrainAccFile = os.path.join(folder, "trainNasTrainAcc_{}.npy".format(str(kth)) )
        trainNasnValAccFile = os.path.join( folder,"trainNasValAcc_{}.npy".format(str(kth)) )
        testAccFile = os.path.join(folder, "testAcc_{}.npy".format(str(kth)) )
        
        accRecord = {"train": np.load(trainNasTrainAccFile),
            "val": np.load(trainNasnValAccFile),
            "test": np.load(testAccFile)
            }
        plot_acc_curve(accRecord, "acc_"+str(kth), "./plot")
