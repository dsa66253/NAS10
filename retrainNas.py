from __future__ import print_function
import math
import os
import sys
from unicodedata import east_asian_width
import torch
import torch.optim as optim
from test import TestController
# import torch.backends.cudnn as cudnn
import argparse
from torch import nn
from torchvision import transforms, datasets
from data.config import cfg_newnasmodel, trainDataSetFolder
from tensorboardX import SummaryWriter
import numpy as np
from data.config import folder
from feature.normalize import normalize
from feature.make_dir import makeDir
from feature.split_data import split_data
from feature.random_seed import set_seed_cpu
from PIL import ImageFile
from tqdm import tqdm
from models.retrainModel import NewNasModel
from alexnet.alexnet import Baseline
from utility.AccLossMonitor import AccLossMonitor
from feature.utility import setStdoutToFile, setStdoutToDefault
from feature.utility import getCurrentTime, accelerateByGpuAlgo, get_device
import matplotlib.pyplot as plt
from utility.DatasetHandler import DatasetHandler
from torchvision import transforms
from  utility.DatasetReviewer import DatasetReviewer
import json 
from utility.HistDrawer import HistDrawer
# from train_nas_5cell import prepareDataloader
stdoutTofile = True
accelerateButUndetermine = cfg_newnasmodel["cuddbenchMark"]
recover = False
def printNetGrad(net):
    for name, para in net.named_parameters():
        print("grad", name, "\n", para)
        break
def parse_args(k):
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('--network', default='newnasmodel', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--genotype_file', type=str, default='genotype_' + str(k) + '.npy',
                        help='put decode file')
    parser.add_argument('--pltSavedDir', type=str, default='./plot',
                        help='plot train loss and val loss')
    args = parser.parse_args()
    return args


def saveAccLoss(kth, lossRecord, accRecord):
    print("save record to ", folder["accLossDir"])
    try:
        np.save(os.path.join(folder["accLossDir"], "retrainTrainLoss_"+str(kth)), lossRecord["train"])
        np.save(os.path.join(folder["accLossDir"], "retrainValnLoss_"+str(kth)), lossRecord["val"])
        np.save(os.path.join(folder["accLossDir"], "retrainTrainAcc_"+str(kth)), accRecord["train"])
        np.save(os.path.join(folder["accLossDir"], "retrainValAcc_"+str(kth)), accRecord["val"])
    except Exception as e:
        print("Fail to save acc and loss")
        print(e)
def prepareDataSet():
    #info prepare dataset
    datasetHandler = DatasetHandler(trainDataSetFolder, cfg, seed_img)
    datasetHandler.addAugmentDataset(transforms.RandomHorizontalFlip(p=1))
    # datasetHandler.addAugmentDataset(transforms.RandomRotation(degrees=10))
    print("training dataset set size:", len(datasetHandler.getTrainDataset()))
    print("val dataset set size:", len(datasetHandler.getValDataset()))
    
    return datasetHandler.getTrainDataset(), datasetHandler.getValDataset()

def prepareDataLoader(trainData, valData):
    #info prepare dataloader
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valData, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    return train_loader, val_loader

def prepareLossFunction():
    #info prepare loss function
    print('Preparing loss function...')
    return  nn.CrossEntropyLoss()

def prepareModel(kth):
    #info load decode json
    filePath = os.path.join(folder["decode"], "{}th_decode.json".format(kth))
    f = open(filePath)
    archDict = json.load(f)
        
    #info prepare model
    print("Preparing model...")
    net = NewNasModel(cellArch=archDict)
    net.train()
    net = net.to(device)
    print("net.cellArch:", net.cellArch)
    print("net", net)
    return net
def prepareOpt(net):
    return optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum,
                    weight_decay=weight_decay)  # 是否采取 weight_decay

def saveCheckPoint(kth, epoch, optimizer, net, lossRecord, accReocrd):
    makeDir(folder["savedCheckPoint"])
    print("save check point kth {} epoch {}".format(kth, epoch))
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': lossRecord,
            'acc': accReocrd
            }, 
            os.path.join(folder["savedCheckPoint"], "{}_{}_{}.pt".format(args.network, kth, epoch)))
    except Exception as e:
        print("Failt to save check point")
        print(e)
        
def recoverFromCheckPoint(model, optimizer):
    pass
    checkpoint = torch.load(folder["savedCheckPoint"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def printNetWeight(net):
    for name, para in net.named_parameters():
        print(name, "\n", para.data)
        
def makeAllDir():
    for folderName in folder:
        print("making folder ", folder[folderName])
        makeDir(folder[folderName])
        
def compareNet(alexnet, net):
    # _, alexPara = alexnet.named_parameters()
    # _, nasPara = net.named_parameters()
    # print(alexPara)
    for alexnet, nasNet in zip(alexnet.named_parameters(), net.named_parameters()):
        alexnetLayerName, alexnetLayerPara = alexnet
        nasNetLayerName , nasNetLyaerPara = nasNet
        print(alexnetLayerName, nasNetLayerName)
        print(alexnetLayerPara.data.sum(), nasNetLyaerPara.data.sum())
        
        
def tmpF(net):
    for netLayerName , netLyaerPara in net.named_parameters():
        print(netLayerName)
        print(netLyaerPara.data.sum())
        break
def weightCount(net):
    count = 0
    for netLayerName , netLyaerPara in net.named_parameters():
        print(netLyaerPara.device)
        shape = netLyaerPara.shape
        dim=1
        for e in shape:
            dim = e*dim
        count = count + dim
    return count
def gradCount(net):
    count = 0
    for netLayerName , netLyaerPara in net.named_parameters():
        if netLyaerPara.grad!=None:
            shape = netLyaerPara.grad.shape
            dim=1
            for e in shape:
                dim = e*dim
            count = count + dim
    return count
def myTrain(kth, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion, writer):
    global last_epoch_val_acc #?幹嘛用
    ImageFile.LOAD_TRUNCATED_IMAGES = True#* avoid damage image file


    # print("Training with learning rate = %f, momentum = %f, lambda = %f " % (initial_lr, momentum, weight_decay))
    #info other setting
    
    
    epoch_size = math.ceil(len(trainData) / batch_size)#* It should be number of batch per epoch
    max_iter = cfg['epoch'] * epoch_size #* it's correct here. It's the totoal iterations.
    #* an iteration go through a mini-batch(aka batch)
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0
    start_iter = 0
    epoch = 0
    # other setting
    print("start to train...")
    
    record_train_loss = np.array([])
    record_val_loss = np.array([])
    record_train_acc = np.array([])
    record_val_acc = np.array([])
    record_test_acc = np.array([])

    #info start training loop
    for iteration in tqdm(range(start_iter, max_iter), unit =" iterations on {}".format(kth)):
        #info things need to do per epoch
        if iteration % epoch_size == 0:
            if (iteration != 0):
                epoch = epoch + 1
            else:
                if recover:
                    net, model_optimizer, epoch, lossRecord, accRecord = recoverFromCheckPoint(kth, epoch, net, model_optimizer)
            
            print("start training epoch{}...".format(epoch))
            
            train_batch_iterator = iter(trainDataLoader)
            
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        train_images, train_labels = next(train_batch_iterator)
        val_batch_iterator = iter(valDataLoader)
        val_images, val_labels = next(val_batch_iterator)
        
        train_images = train_images.to(device)
        train_labels = train_labels.to(device)
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        # print("================111")
        model_optimizer.zero_grad(set_to_none=True)
        #info forward pass
        train_outputs = net(train_images)
        #info calculate loss
        train_loss = criterion(train_outputs, train_labels)
        #info backward pass
        # print(torch.cuda.memory_allocated(device=device))
        # print(torch.cuda.memory_summary(device=device) )
        # print("weightCount ", weightCount(net))
        # print("gradCount ", gradCount(net))
        train_loss.backward()
        # print(torch.cuda.memory_summary(device=device) )
        # print("weightCount ", weightCount(net))
        # print("gradCount ", gradCount(net))
        # exit()
        # while True:
        #     pass
        #info update weight
        model_optimizer.step()

        # printNetGrad(net)
        # printNetWeight(net)
        # print("================")
        #info do statistics after this epoch
        if iteration % epoch_size == 0:
            with torch.no_grad():
                #info validation data forward pass
                val_outputs = net(val_images)
                _, predicts_val = torch.max(val_outputs.data, 1)
                val_loss = criterion(val_outputs, val_labels)
                record_val_loss = np.append(record_val_loss, val_loss.item())
            
            #info calculate training data acc and loss 
            _, predicts = torch.max(train_outputs.data, 1)
            record_train_loss = np.append(record_train_loss, train_loss.item())
            total_images = 0
            correct_images = 0
            total_images += train_labels.size(0)
            correct_images += (predicts == train_labels).sum()
            trainAcc = correct_images / total_images * 100
            record_train_acc = np.append(record_train_acc, trainAcc.cpu())
            
            #info calculate validation data acc and loss 
            total_images_val = 0
            correct_images_val = 0
            total_images_val += val_labels.size(0)
            correct_images_val += (predicts_val == val_labels).sum()
            valAcc = correct_images_val / total_images_val *100
            record_val_acc = np.append(record_val_acc, valAcc.cpu())
            
            #info record acc an loss
            # writer.add_scalar('Train_Loss/k='+str(kth), train_loss.item(), epoch)
            # writer.add_scalar('Val_Loss/k='+str(kth), val_loss.item(), epoch)
            # writer.add_scalar('train_Acc/k='+str(kth), trainAcc, epoch)
            # writer.add_scalar('val_Acc/k='+str(kth), valAcc, epoch)
            last_epoch_val_acc = 100 * correct_images_val / total_images_val
        # exit()
        # if iteration>=10:
        #     break

    lossRecord = {"train": record_train_loss, "val": record_val_loss}
    accRecord = {"train": record_train_acc, "val": record_val_acc}
    torch.save(net.state_dict(), os.path.join(folder["retrainSavedModel"], cfg['name'] + str(kth) + '_Final.pt'))
    return last_epoch_val_acc, lossRecord, accRecord


if __name__ == '__main__':
    device = get_device()
    torch.device(device)
    print("running on device: {}".format(device))
    torch.set_printoptions(precision=6, sci_mode=False, threshold=1000)
    torch.set_default_dtype(torch.float32) #* torch.float will slow the training speed
    valList = []

    for k in range(0, 3):
        #info handle stdout to a file
        if stdoutTofile:
            f = setStdoutToFile(folder["log"]+"/retrain_5cell_{}th.txt".format(str(k)))
        
        #info set seed
        if k == 0:
            seed_img = 10
            seed_weight = 20
        elif k == 1:
            seed_img = 255
            seed_weight = 278
        else:
            seed_img = 830
            seed_weight = 953
            
        accelerateByGpuAlgo(cfg_newnasmodel["cuddbenchMark"])
        set_seed_cpu(seed_weight)
        #! test same initial weight
        args = parse_args(str(k))
        makeAllDir()
        

        cfg = None
        if args.network == "newnasmodel":
            cfg = cfg_newnasmodel
        else:
            print('Retrain Model %s doesn\'t exist!' % (args.network))
            sys.exit(0)
            
        batch_size = cfg['batch_size']

        #todo find what do these stuff do
        num_workers = args.num_workers
        momentum = args.momentum
        weight_decay = args.weight_decay
        initial_lr = args.lr
        gamma = args.gamma

        
        #info test
        test = TestController(cfg, device)
        # writer = SummaryWriter(log_dir=folder["tensorboard_retrain"], comment="{}th".format(str(k)))
        
        print("seed_img{}, seed_weight{} start at ".format(seed_img, seed_weight), getCurrentTime())
        print("cfg", cfg)
        
        #info training process 
        trainData, valData = prepareDataSet()
        trainDataLoader, valDataLoader = prepareDataLoader(trainData, valData)
        criterion = prepareLossFunction()
        net = prepareModel(k)
        histDrawer = HistDrawer(folder["pltSavedDir"])
        histDrawer.drawNetConvWeight(net, tag="ori_{}".format(str(k)))
        model_optimizer = prepareOpt(net)
        
        last_epoch_val_ac, lossRecord, accRecord = myTrain(k, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion, writer=None)  # 進入model訓練
        histDrawer.drawNetConvWeight(net, tag="trained_{}".format(str(k)))
        #info record training processs
        alMonitor = AccLossMonitor(k, folder["pltSavedDir"], folder["accLossDir"], trainType="retrain")
        alMonitor.plotAccLineChart(accRecord)
        alMonitor.plotLossLineChart(lossRecord)
        alMonitor.saveAccLossNp(accRecord, lossRecord)

        valList.append(last_epoch_val_ac)
        print('retrain validate accuracy:', valList)
        
        # writer.close()
        #info handle output file
        if stdoutTofile:
            setStdoutToDefault(f)
            
    print('retrain validate accuracy:', valList)



