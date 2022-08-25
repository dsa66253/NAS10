import datetime
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
def accelerateByGpuAlgo(setTo):
    print("accelerate by gpu algo but got undertermine result mode: ", setTo)
    #! set to false can reproduce training result
    #! while set to true will fasten training process but to gaurantee same training result
    torch.backends.cudnn.benchmark = setTo
    torch.backends.cudnn.deterministic = not setTo
    
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        print("\033[1;31m No Gpu available  \n")
        return "cpu"
        # exit(-1)

def setStdoutToFile(filePath):
    print("std output to ", filePath)
    f = open(filePath, 'w')
    sys.stdout = f
    return f

def setStdoutToDefault(f):
    f.close()
    sys.stdout = sys.__stdout__

def getCurrentTime():
    # return string type of current date and time
    loc_dt = datetime.datetime.today() 
    loc_dt_format = loc_dt.strftime("%Y/%m/%d %H:%M:%S")
    return loc_dt_format

def getCurrentTime1():
    # return string type of current date and time
    loc_dt = datetime.datetime.today() 
    loc_dt_format = loc_dt.strftime("%H_%M_%S")
    return loc_dt_format


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
    fig, ax = plt.subplots()
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