from __future__ import print_function
import math
import os
import sys
import torch
import torch.optim as optim
# import torch.backends.cudnn as cudnn
import argparse
from torch import nn
from torchvision import datasets
from data.config import cfg_newnasmodel, trainDataSetFolder, cfg_alexnet
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
from datasetPractice import DatasetHandler
from torchvision import transforms
from  DatasetReviewer import DatasetReviewer
import matplotlib.pyplot as plt
stdoutTofile = True
accelerateButUndetermine = cfg_newnasmodel["cuddbenchMark"]
def parse_args(k):
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('--network', default='NewNasModel', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--genotype_file', type=str, default='genotype_' + str(k) + '.npy',
                        help='put decode file')
    args = parser.parse_args()
    return args

def makeAllDir():
    for folderName in folder:
        # print("making folder ", folder[folderName])
        makeDir(folder[folderName])
        
def prepareDataSet():
    #info prepare dataset
    datasetHandler = DatasetHandler(trainDataSetFolder, cfg, seed_img)
    # datasetHandler.addAugmentDataset(transforms.RandomHorizontalFlip(p=1))
    # datasetHandler.addAugmentDataset(transforms.RandomRotation(degrees=10))
    print("training dataset set size:", len(datasetHandler.getTrainDataset()))
    print("val dataset set size:", len(datasetHandler.getValDataset()))
    
    return datasetHandler.getTrainDataset(), datasetHandler.getValDataset()

def prepareDataLoader(trainData, valData):
    #info prepare dataloader
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=cfg["batch_size"], num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valData, batch_size=cfg["batch_size"], num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    return train_loader, val_loader
def prepareLossFunction():
    #info prepare loss function
    print('Preparing loss function...')
    return  nn.CrossEntropyLoss()

def prepareOpt(net):
    return optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum,
                    weight_decay=weight_decay)  # 是否采取 weight_decay

def prepareModel():
    
    if args.network == "NewNasModel":
        #info check decode alphas and load it
        if os.path.isdir(folder["decode_folder"]):
            try:
                
                genotype_filename = os.path.join(folder["decode_folder"], args.genotype_file)
                
                cell_arch = np.load(genotype_filename)
                print("successfully load decode alphas")
            except Exception as e:
                print("Fail to load decode alpha:", e)
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
            # print("net.cellArchTrans:", net.cellArchTrans)
            print("net", net)
        return net
    elif cfg["name"] == "alexnet":
        # alexnet model
        net = Baseline(cfg["numOfClasses"])
        print("net", net)
        net = net.to(device)
        net.train()
        return net
    else:
        print("cannot prepare net")
        exit()

def tmpF(net):
    for netLayerName , netLyaerPara in net.named_parameters():
        print(netLayerName)
        print(netLyaerPara.data)
        break
        fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
        bin = ax.hist(np.reshape(netLyaerPara.data.cpu().numpy(), (-1)), bins=50)
        # # print(bin[0], len(bin[0]))
        # # print(bin[1], len(bin[1]))
        # # print(bin[2], len(bin[2]))
        plt.savefig("./hist")
        exit()

        
def printNetGrad(net):
    for name, para in net.named_parameters():
        print("gradient", name, "\n", para.grad)
        break
def printNetWeight(net):
    for name, para in net.named_parameters():
        print(name, "\n", para.data)
def saveAccLoss(kth, lossRecord, accRecord):
    print("save record to ", folder["accLossDir"])
    try:
        np.save(os.path.join(folder["accLossDir"], "retrainTrainLoss_"+str(kth)+cfg["name"]), lossRecord["train"])
        np.save(os.path.join(folder["accLossDir"], "retrainValnLoss_"+str(kth)+cfg["name"]), lossRecord["val"])
        np.save(os.path.join(folder["accLossDir"], "retrainTrainAcc_"+str(kth)+cfg["name"]), accRecord["train"])
        np.save(os.path.join(folder["accLossDir"], "retrainValAcc_"+str(kth)+cfg["name"]), accRecord["val"])
    except Exception as e:
        print("Fail to save acc and loss")
        print(e)
        
def train(kth, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion, writer):
    last_epoch_val_acc = 0
    epoch_size = math.ceil(len(trainData) / cfg["batch_size"])#* It should be number of batch per epoch
    numOfIter = cfg['epoch'] * epoch_size 
    
    epoch = 0
    record_train_loss = np.array([])
    record_val_loss = np.array([])
    record_train_acc = np.array([])
    record_val_acc = np.array([])
    for iteration in tqdm(range(numOfIter), unit=" iters on{}".format(kth)):
        #info things need to do per epoch
        if iteration % epoch_size == 0:
            print("start training epoch{}...".format(epoch))
            if (iteration != 0):
                epoch = epoch + 1
            accRecord = {"train": record_train_acc, "val": record_val_acc}
            lossRecord = {"train": record_train_loss, "val": record_val_loss}
            # saveCheckPoint(kth, epoch, model_optimizer, net, lossRecord, accRecord)
            train_batch_iterator = iter(trainDataLoader)
            
        #info prepare data
        train_images, train_labels = next(train_batch_iterator)
        val_batch_iterator = iter(valDataLoader)
        val_images, val_labels = next(val_batch_iterator)
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        
        # print("iteration", iteration)
        # forward pass
        train_outputs = net(train_images)

        # calculate loss
        train_loss = criterion(train_outputs, train_labels)
        # backward pass
        model_optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        # update weight

        # printNetGrad(net)
        model_optimizer.step()
        # print()
        # if iteration%10==0:
        #     tmpF(net)
        
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
    lossRecord = {"train": record_train_loss, "val": record_val_loss}
    accRecord = {"train": record_train_acc, "val": record_val_acc}
    torch.save(net.state_dict(), os.path.join(folder["retrainSavedModel"], cfg['name'] + str(kth) + '_Final.pth'))
    return last_epoch_val_acc, lossRecord, accRecord
        
        
        
if __name__ == '__main__':
    device = get_device()
    print("running on device: {}".format(device))
    accelerateByGpuAlgo(cfg_newnasmodel["cuddbenchMark"])
    torch.set_printoptions(precision=6, sci_mode=False, threshold=1000)
    torch.set_default_dtype(torch.float32) #* torch.float will slow the training speed
    valList = []
    for k in range(3):
        #info set seed
        if k == 0:
            seed_img = 10
            seed_weight = 20
        elif k == 1:
            seed_img = 255
            seed_weight = 278
        elif k==2:
            seed_img = 830
            seed_weight = 953   
        elif k==3:
            seed_img = 1000
            seed_weight = 1100   
            
        elif k==4:
            seed_img = 1200
            seed_weight = 1300   
            
        elif k==5:
            seed_img = 1400
            seed_weight = 1500 

        set_seed_cpu(seed_weight)
        
        
        #! test same initial weight
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        args = parse_args(str(k))
        
        cfg = None
        if args.network == "NewNasModel":
            cfg = cfg_newnasmodel
        elif args.network=="alexnet":
            cfg = cfg_alexnet
        else:
            print("no cfg: ", args.network)
            sys.exit(0)
        
        
        if stdoutTofile:
            f = setStdoutToFile(folder["log"]+"/{}_{}th.txt".format(cfg["name"], str(k)))
        makeAllDir()
        print("seed_img{}, seed_weight{} start at ".format(seed_img, seed_weight), getCurrentTime())
        print("cfg", cfg)
        
        #todo find what do these stuff do
        num_workers = args.num_workers
        momentum = args.momentum
        weight_decay = args.weight_decay
        initial_lr = args.lr
        gamma = args.gamma
        
        trainData, valData = prepareDataSet()
        trainDataLoader, valDataLoader = prepareDataLoader(trainData, valData)
        criterion = prepareLossFunction()
        set_seed_cpu(seed_weight)
        net = prepareModel()
        model_optimizer = prepareOpt(net)
        tensorboardDir = os.path.join(folder["tensorboard_retrain"], cfg["name"])
        makeDir(tensorboardDir)
        writer = SummaryWriter(log_dir=tensorboardDir, comment="{}th".format(str(k)))
        
        # tmpF(net)
        last_epoch_val_ac, lossRecord, accRecord = train(k, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion, writer)  # 進入model訓練
        # tmpF(net)
        #info record training processs
        plot_loss_curve(lossRecord, "loss_{}".format(k), folder["pltSavedDir"])
        plot_acc_curve(accRecord, "acc_{}".format(k), folder["pltSavedDir"])
        saveAccLoss(k, lossRecord, accRecord)
        valList.append(last_epoch_val_ac)
        print('retrain validate accuracy:', valList)
        
        
        # print('retrain validate accuracy:', valList)
        # datasetReviewer = DatasetReviewer(cfg["batch_size"],
        #                                 k,
        #                                 DatasetHandler.getOriginalDataset(trainDataSetFolder, cfg, seed_img), 
        #                                 device)
        # datasetReviewer.makeSummary(trainDataLoader, writer, net)
        # datasetReviewer.showReport()

        writer.close()
        #info handle output file
        if stdoutTofile:
            setStdoutToDefault(f)
            