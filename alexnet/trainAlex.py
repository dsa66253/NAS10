import os
import sys
import torch
import argparse
import torch.optim as optim
from test import TestController
import torch.nn as nn
import math
import time
from torchvision import datasets
from data.config import cfg_nasmodel, cfg_alexnet, trainDataSetFolder
from alexnet.alexnet import Baseline
# from tensorboardX import SummaryWriter #* how about use tensorbaord instead of tensorboardX
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from data.config import folder
from feature.normalize import normalize
from feature.make_dir import makeDir
from feature.split_data import split_data
from feature.random_seed import set_seed_cpu
from PIL import ImageFile
from tqdm import tqdm
from model import Model
from data.config import epoch_to_drop
from feature.utility import getCurrentTime, setStdoutToDefault, setStdoutToFile, accelerateByGpuAlgo
from feature.utility import plot_acc_curve, plot_loss_curve, get_device
from utility.alphasMonitor import AlphasMonitor
from  DatasetReviewer import DatasetReviewer
stdoutTofile = True
accelerateButUndetermine = False
recover = False

def printNetGrad(net):
    for name, para in net.named_parameters():
        print("grad", name, "\n", para)
        break

def parse_args():
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('--network', default='alexnet', help='Backbone network alexnet or nasmodel')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--nas_lr', '--nas-learning-rate', default=3e-3, type=float,
                        help='initial learning rate for nas optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--trainDataSetFolder', default='./dataset1/train',
                    help='training data set folder')
    args = parser.parse_args()
    return args

def prepareDataSet():
    #info prepare dataset
    print('Loading Dataset from {} with seed_img {}'.format(trainDataSetFolder, str(seed_img)))
    train_transforms = normalize(seed_img, img_dim)  # 正規化照片
    try:
        all_data = datasets.ImageFolder(trainDataSetFolder, transform=train_transforms)
        print("all_data.class_to_idx", all_data.class_to_idx)
    except Exception as e:
        print("Fail to load data set from: ",  trainDataSetFolder)
        print(e)
        exit()
    train_data, val_data = split_data(all_data, 0.2)  # 切訓練集跟驗證集
    return all_data, train_data, val_data

def prepareDataLoader(trainData, valData):
    #info prepare dataloader
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valData, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    return train_loader, val_loader

def prepareLossFunction():
    print('Preparing loss function...')
    return  nn.CrossEntropyLoss()

def prepareModel():
    #info prepare model
    print("Preparing model...")
    if cfg['name'] == 'alexnet':
        # alexnet model
        net = Baseline(cfg["numOfClasses"])
        print("net", net)
        net = net.to(device)
        net.train()
    elif cfg['name'] == 'NasModel':
        # nas model
        # todo why pass no parameter list to model, and we got cfg directly in model.py from config.py
        net = Model()
        print("net", net)
        #! move to cuda before assign net's parameters to optim, otherwise, net on cpu will slow down training speed
        net = net.to(device)
        net.train()
    return net

def prepareOpt(net):
    #info prepare optimizer
    print("Preparing optimizer...")
    if cfg['name'] == 'alexnet':  # BASELINE
        optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        return optimizer
    
def printNetWeight(net):
    for name, para in net.named_parameters():
        print(name, para)


    
def saveCheckPoint(kth, epoch, optimizer, net, lossRecord, accReocrd):
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
def recoverFromCheckPoint(kth, epoch, model, optimizer):
    checkpoint = torch.load(os.path.join(folder.savedCheckPoint, "{}_{}_{}.pt".format(args.network, kth, epoch)))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("recover from check point at epoch ", checkpoint['epoch'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def saveAccLoss(kth, lossRecord, accRecord):
    print("save record to ", folder["accLossDir"])
    try:
        np.save(os.path.join(folder["accLossDir"], "trainNasTrainLoss_"+str(kth)), lossRecord["train"])
        np.save(os.path.join(folder["accLossDir"], "trainNasValnLoss_"+str(kth)), lossRecord["val"])
        np.save(os.path.join(folder["accLossDir"], "trainNasTrainAcc_"+str(kth)), accRecord["train"])
        np.save(os.path.join(folder["accLossDir"], "trainNasValAcc_"+str(kth)), accRecord["val"])
        np.save(os.path.join(folder["accLossDir"], "trainNasTestAcc_"+str(kth)), accRecord["test"])
    except Exception as e:
        print("Fail to save acc and loss")
        print(e)
def makeAllDir():
    for folderName in folder:
        print("making folder ", folder[folderName])
        makeDir(folder[folderName])
        
def tmpF(net):
    for netLayerName , netLyaerPara in net.named_parameters():
        print(netLayerName)
        print(netLyaerPara.data.sum())

        
        
def myTrain(kth, trainData, train_loader, val_loader, net, model_optimizer, criterion, writer):
    
    # calculate how many iterations
    epoch_size = math.ceil(len(trainData) / batch_size)#* It should be number of batch per epoch
    max_iter = max_epoch * epoch_size #* it's correct here. It's the totoal iterations.
    #* an iteration go through a mini-batch(aka batch)
    start_iter = 0
    epoch = 0
    
    # other setting
    # writer = SummaryWriter()
    # writer = SummaryWriter(comment="{}_{}th".format(cfg["name"], kth))
    print("start to train...")
    
    record_train_loss = np.array([])
    record_val_loss = np.array([])
    record_train_acc = np.array([])
    record_val_acc = np.array([])
    record_test_acc = np.array([])
    
    print("minibatch size: ", epoch_size)
    #info start training loop
    for iteration in tqdm(range(start_iter, max_iter), unit =" iterations on {}".format(kth)):
        
        #info things need to do per epoch
        if iteration % epoch_size == 0:
            if (iteration != 0):
                epoch = epoch + 1
            print("start training epoch{}...".format(epoch))
            
            accRecord = {"train": record_train_acc, "val": record_val_acc, "test": record_test_acc}
            lossRecord = {"train": record_train_loss, "val": record_val_loss}

            train_batch_iterator = iter(train_loader)
        
        

        # load train data
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        train_images, train_labels = next(train_batch_iterator)
        val_batch_iterator = iter(val_loader)
        val_images, val_labels = next(val_batch_iterator)

        train_images = train_images.to(device)
        train_labels = train_labels.to(device)
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        # print("train_labels","\n", train_labels)
        model_optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        
        train_outputs = net(train_images)
        printNetGrad(net)
        # calculate loss
        train_loss = criterion(train_outputs, train_labels)
        # backward pass
        train_loss.backward()
        # print("epoch >= cfg['start_train_nas_epoch']", epoch >= cfg['start_train_nas_epoch'])
        # print("(epoch - cfg['start_train_nas_epoch']) % 2 == 0", (epoch - cfg['start_train_nas_epoch']) % 2 == 0)
        # print("epoch", epoch, "cfg['start_train_nas_epoch']", cfg['start_train_nas_epoch'])
        # take turns to optimize weight and alphas
        model_optimizer.step()
        print("iteration", iteration)
        tmpF(net)
        #info recording training process
        # model預測出來的結果 (訓練集)
        _, predicts = torch.max(train_outputs.data, 1)
        # record_train_loss.append(train_loss.item())
        
        
        #! Why she use validation directly at the end of an iteration.
        #! Usually we use validation after finishing all training.
        #! And chose the model generate with best accuracy on validation set
        #info after training this epoch
        if (iteration % epoch_size == 0):
            val_outputs = net(val_images)
            _, predicts_val = torch.max(val_outputs.data, 1)
            record_train_loss = np.append(record_train_loss, train_loss.item())
            val_loss = criterion(val_outputs, val_labels)
            record_val_loss = np.append(record_val_loss, val_loss.item())
            # print("end iteration", iteration, " net.Alphas", net.alphas)
            # 計算訓練集準確度
            total_images = 0
            correct_images = 0
            total_images += train_labels.size(0)
            correct_images += (predicts == train_labels).sum()
            # 計算驗證集準確度
            total_images_val = 0
            correct_images_val = 0
            total_images_val += val_labels.size(0)
            correct_images_val += (predicts_val == val_labels).sum()

            
            #info test set
            print("start test epoch", epoch)
            testAcc = test.test(net)
            # testAcc = torch.tensor(0)

            record_test_acc = np.append(record_test_acc, testAcc.cpu())
            # print(
            #     'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || train_loss: {:.4f} || val_loss: {:.4f} || train_Accuracy: {:.4f} || val_Accuracy: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || '
            #     'ETA: {} '
            #         .format(epoch, max_epoch, (iteration % epoch_size) + 1,
            #                 epoch_size, iteration + 1, max_iter, train_loss.item(), val_loss.item(),
            #                 100 * correct_images.item() / total_images,
            #                 100 * correct_images_val.item() / total_images_val,
            #                 0.02, batch_time, str(datetime.timedelta(seconds=eta))))

            #info 使用tensorboard紀錄LOSS、ACC            
            trainAcc = correct_images / total_images
            valAcc = correct_images_val / total_images_val
            record_train_acc = np.append(record_train_acc, trainAcc.cpu())
            record_val_acc = np.append(record_val_acc, valAcc.cpu())
            
            writer.add_scalar('Train_Loss/k='+str(kth), train_loss.item(), epoch)
            writer.add_scalar('Val_Loss/k='+str(kth), val_loss.item(), epoch)
            writer.add_scalar('train_Acc/k='+str(kth), 100 * trainAcc, epoch)
            writer.add_scalar('val_Acc/k='+str(kth), 100 * valAcc, epoch)
            writer.add_scalar('test_Acc/k='+str(kth), 100 * testAcc, epoch)
            accToTensorBoard = {
                "trainAcc": 100 * trainAcc,
                "valAcc": 100 * valAcc,
                "testAcc": 100 * testAcc,
            }
            writer.add_scalars('Acc/k='+str(kth), accToTensorBoard, epoch)
            last_epoch_val_acc = 100 * correct_images_val / total_images_val
            
            #info save checkpoint model
            accRecord = {"train": record_train_acc, "val": record_val_acc, "test": record_test_acc}
            lossRecord = {"train": record_train_loss, "val": record_val_loss}
            saveCheckPoint(kth, epoch, model_optimizer, net, lossRecord, accRecord)

        if iteration>=10:
            break

    lossRecord = {"train": record_train_loss, "val": record_val_loss}
    accRecord = {"train": record_train_acc, "val": record_val_acc, "test": record_test_acc}


    torch.save(net.state_dict(), os.path.join(folder["savedCheckPoint"], cfg['name'] + str(kth) + '_Final.pt'))
    writer.close()
    
    return last_epoch_val_acc, lossRecord, accRecord





if __name__ == '__main__':
    device = get_device()
    torch.device(device)
    print("running on device: {}".format(device))
    torch.set_printoptions(precision=6, sci_mode=False, threshold=1000)
    torch.set_default_dtype(torch.float32) #* torch.float will slow the training speed
    valList = []
    
    for k in range(3):
        #info set stdout to file
        if stdoutTofile:
            f = setStdoutToFile( os.path.join( folder["log"], "train_nas_5cell_{}th.txt".format(str(k)) ) )
        #info diifferent seeds fro different initail weights
        if k == 0:
            seed_img = 10
            seed_weight = 20
        elif k == 1:
            seed_img = 255
            seed_weight = 278
        else:
            seed_img = 830
            seed_weight = 953
        
        accelerateByGpuAlgo(accelerateButUndetermine)
        set_seed_cpu(seed_weight)  # 控制照片批次順序
        
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        args = parse_args()
        
        makeAllDir()
        
        cfg = None
        if args.network == "alexnet":
            cfg = cfg_alexnet
        else:
            print('Model {} doesn\'t exist!'.format(args.network))
            sys.exit(0)

        img_dim = cfg['image_size']
        num_gpu = cfg['ngpu']
        batch_size = cfg['batch_size']
        max_epoch = cfg['epoch']
        gpu_train = cfg['gpu_train']

        num_workers = args.num_workers
        momentum = args.momentum
        weight_decay = args.weight_decay
        initial_lr = args.lr
        nas_initial_lr = args.nas_lr
        gamma = args.gamma

        #info test
        test = TestController(cfg, device)
        writer = SummaryWriter(log_dir=folder["tensorboard_trainNas"], comment="{}th".format(str(k)))
        
        print("seed_img {}, seed_weight {} start at ".format(seed_img, seed_weight), getCurrentTime())
        print("training cfg", cfg)
        allData, trainData, valData = prepareDataSet()
        trainDataLoader, valDataLoader = prepareDataLoader(trainData, valData)
        criterion = prepareLossFunction()
        net = prepareModel()
        tmpF(net)
        model_optimizer = prepareOpt(net)
        
        
        last_epoch_val_ac, lossRecord, accRecord  = myTrain(k, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion, writer)  # 進入model訓練
        saveAccLoss(k, lossRecord, accRecord)
        plot_loss_curve(lossRecord, "loss_{}".format(k), folder["pltSavedDir"])
        plot_acc_curve(accRecord, "acc_{}".format(k), folder["pltSavedDir"])
        valList.append(last_epoch_val_ac)
        print('train validate accuracy:', valList)
        
        datasetReviewer = DatasetReviewer(batch_size, k, allData, device)
        datasetReviewer.makeSummary(trainDataLoader, writer, net)
        datasetReviewer.showReport()
        
        writer.close()
        tmpF(net)
        if stdoutTofile:
            setStdoutToDefault(f)
        
        exit() #* for examine why same initial value will get different trained model
    print('train validate accuracy:', valList)
        
        
        
        
        
        
        
        
        
        
        