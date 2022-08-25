from __future__ import print_function
import math
import os
import sys
import torch
import torch.optim as optim
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
from retrainModel import NewNasModel
from alexnet import Baseline
from feature.utility import plot_acc_curve, setStdoutToFile, setStdoutToDefault
from feature.utility import getCurrentTime, accelerateByGpuAlgo, get_device, plot_loss_curve
import matplotlib.pyplot as plt
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
    print('Loading Dataset from {} with seed_img {}'.format(trainDataSetFolder, str(seed_img)))
    train_transforms = normalize(seed_img, cfg['image_size'])  # 正規化照片
    try:
        all_data = datasets.ImageFolder(trainDataSetFolder, transform=train_transforms)
        print("all_data.class_to_idx", all_data.class_to_idx)
    except Exception as e:
        print("Fail to load data set from: ",  trainDataSetFolder)
        print(e)
        exit()
    train_data, val_data = split_data(all_data, 0.2)  # 切訓練集跟驗證集
    return train_data, val_data

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

def prepareModel():
    #info check decode alphas and load it
    if os.path.isdir(folder["decode_folder"]):
        try:
            genotype_filename = os.path.join(folder["decode_folder"], args.genotype_file)
            cell_arch = np.load(genotype_filename)
            print("successfully load decode alphas")
        except:
            print("Fail to load decode alpha")
    else:
        print('decode_folder does\'t exist!')
        sys.exit(0)
        
    #info prepare model
    print("Preparing model...")
    if cfg['name'] == 'NewNasModel':
        # set_seed_cpu(seed_weight) #* have already set seed in main function 
        net = NewNasModel(cellArch=cell_arch)
        net.train()
        net = net.to(device)
        print("net.cellArchTrans:", net.cellArchTrans)
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

        
def myTrain(kth, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion):
    print("start train kth={}...".format(kth))
    global last_epoch_val_acc #?幹嘛用
    ImageFile.LOAD_TRUNCATED_IMAGES = True#* avoid damage image file


    # print("Training with learning rate = %f, momentum = %f, lambda = %f " % (initial_lr, momentum, weight_decay))
    #info other setting
    writer = SummaryWriter(log_dir=folder["tensorboard_retrain"], comment="{}th".format(str(kth)))
    
    epoch_size = math.ceil(len(trainData) / batch_size)#* It should be number of batch per epoch
    max_iter = cfg['epoch'] * epoch_size #* it's correct here. It's the totoal iterations.
    #* an iteration go through a mini-batch(aka batch)
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0
    start_iter = 0
    epoch = 0
    record_train_loss = np.array([])
    record_val_loss = np.array([])
    record_train_acc = np.array([])
    record_val_acc = np.array([])
    print('Start to train...')

    #info start training loop
    for iteration in tqdm(range(start_iter, max_iter), unit =" iterations on {}".format(kth)):
        
        #info things need to do per epoch
        
        if iteration % epoch_size == 0:
            print("start training epoch{}...".format(epoch))
            if (iteration != 0):
                epoch = epoch + 1
            else:
                if recover:
                    net, model_optimizer, epoch, lossRecord, accRecord = recoverFromCheckPoint(kth, epoch, net, model_optimizer)

            accRecord = {"train": record_train_acc, "val": record_val_acc}
            lossRecord = {"train": record_train_loss, "val": record_val_loss}
            saveCheckPoint(kth, epoch, model_optimizer, net, lossRecord, accRecord)
            train_batch_iterator = iter(trainDataLoader)

        # lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)
        train_images, train_labels = next(train_batch_iterator)
        val_batch_iterator = iter(valDataLoader)
        val_images, val_labels = next(val_batch_iterator)
        # plot_img(train_images, train_labels, val_images, val_labels)
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        # print("train_labels","\n", train_labels)
        
        # forward pass
        train_outputs = net(train_images)
        printNetGrad(net)
        # calculate loss
        train_loss = criterion(train_outputs, train_labels)
        # backward pass
        model_optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        # update weight
        model_optimizer.step()
        print("iteration", iteration)
        tmpF(net)

        if iteration % epoch_size == 0:
            with torch.no_grad():
                #info recording training accruacy
                _, predicts = torch.max(train_outputs.data, 1)
                record_train_loss = np.append(record_train_loss, train_loss.item())

                val_outputs = net(val_images)
                _, predicts_val = torch.max(val_outputs.data, 1)
                val_loss = criterion(val_outputs, val_labels)
                record_val_loss = np.append(record_val_loss, val_loss.item())


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

            trainAcc = correct_images / total_images
            valAcc = correct_images_val / total_images_val
            record_train_acc = np.append(record_train_acc, trainAcc.cpu())
            record_val_acc = np.append(record_val_acc, valAcc.cpu())
            writer.add_scalar('Train_Loss', train_loss.item(), iteration + 1)
            writer.add_scalar('Val_Loss', val_loss.item(), iteration + 1)
            writer.add_scalar('train_Acc', 100 * trainAcc, iteration + 1)
            writer.add_scalar('val_Acc', 100 * valAcc, iteration + 1)
            last_epoch_val_acc = 100 * correct_images_val / total_images_val
        if iteration>=10:
            break
    # images, _ = next(iter(trainDataLoader))
    # writer.add_graph(net, images.to(device))

    lossRecord = {"train": record_train_loss, "val": record_val_loss}
    accRecord = {"train": record_train_acc, "val": record_val_acc}
    torch.save(net.state_dict(), os.path.join(folder["retrainSavedModel"], cfg['name'] + str(kth) + '_Final.pth'))
    return last_epoch_val_acc, lossRecord, accRecord


if __name__ == '__main__':
    device = get_device()
    print("running on device: {}".format(device))
    torch.set_printoptions(precision=6, sci_mode=False, threshold=1000)
    torch.set_default_dtype(torch.float32) #* torch.float will slow the training speed
    valList = []

    for k in range(3):
        #info handle stdout to a file
        if stdoutTofile:
            f = setStdoutToFile(folder["log"]+"/retrain_5cell_{}th.txt".format(str(k)))
            
        accelerateByGpuAlgo(cfg_newnasmodel["cuddbenchMark"])
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
        #! test same initial weight
        ImageFile.LOAD_TRUNCATED_IMAGES = True
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

        set_seed_cpu(seed_weight)
        
        print("seed_img{}, seed_weight{} start at ".format(seed_img, seed_weight), getCurrentTime())
        print("cfg", cfg)
        trainData, valData = prepareDataSet()
        trainDataLoader, valDataLoader = prepareDataLoader(trainData, valData)
        criterion = prepareLossFunction()
        net = prepareModel()
        # alexnet = Baseline(10)f
        # alexnet.train()
        # alexnet.to(device)
        # # printNetWeight(net)
        tmpF(net)
        model_optimizer = prepareOpt(net)
        last_epoch_val_ac, lossRecord, accRecord = myTrain(k, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion)  # 進入model訓練
        
        #info record training processs
        plot_loss_curve(lossRecord, "loss_{}".format(k), folder["pltSavedDir"])
        plot_acc_curve(accRecord, "acc_{}".format(k), folder["pltSavedDir"])
        saveAccLoss(k, lossRecord, accRecord)
        valList.append(last_epoch_val_ac)
        print('retrain validate accuracy:', valList)
        
        #info handle output file
        print('retrain validate accuracy:', valList)
        tmpF(net)
        if stdoutTofile:
            setStdoutToDefault(f)
            
        exit()
    print('retrain validate accuracy:', valList)



