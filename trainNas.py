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
import torchvision.transforms as T
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
from models.mymodel import Model
from data.config import epoch_to_drop
from feature.utility import getCurrentTime, setStdoutToDefault, setStdoutToFile, accelerateByGpuAlgo
from feature.utility import plot_acc_curve, plot_loss_curve, get_device
from utility.alphasMonitor import AlphasMonitor
# from utility.TransformImgTester import TransformImgTester
from utility.DatasetHandler import DatasetHandler
from torchvision import transforms
from utility.DatasetReviewer import DatasetReviewer
from utility.AccLossMonitor import AccLossMonitor
import json
from models.arch import simpleArch
from models.initWeight import initialize_weights
stdoutTofile = True
accelerateButUndetermine = False
recover = False

def parse_args():
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('--network', default='nasmodel', help='Backbone network alexnet or nasmodel')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--nas_lr', '--nas-learning-rate', default=3e-3, type=float,
                        help='initial learning rate for nas optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--trainDataSetFolder', default='../dataset/train',
                    help='training data set folder')
    args = parser.parse_args()
    return args

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
    print('Preparing loss function...')
    return  nn.CrossEntropyLoss()

def prepareModel():
    #info load decode json
    # filePath = os.path.join("models", "simpleArch.json")
    # f = open(filePath)
    # archDict = json.load(f)

    #info prepare model
    print("Preparing model...")
    if cfg['name'] == 'alexnet':
        # alexnet model
        net = Baseline(cfg["numOfClasses"])
        net = net.to(device)
        net.train()
    elif cfg['name'] == 'NasModel':
        # nas model
        # todo why pass no parameter list to model, and we got cfg directly in model.py from config.py
        net = Model(simpleArch)
        print("net", net)
        #! move to cuda before assign net's parameters to optim, otherwise, net on cpu will slow down training speed
        net = net.to(device)
        net.train()
    initialize_weights(net, seed_img) #! compare with initialize_weights
    return net

def prepareOpt(net):
    #info prepare optimizer
    print("Preparing optimizer...")
    if cfg['name'] == 'alexnet':  # BASELINE
        optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        return optimizer
    elif cfg['name'] == 'NasModel':
        model_optimizer = optim.SGD(net.getWeight(), lr=initial_lr, momentum=momentum,
                                    weight_decay=weight_decay)
        nas_optimizer = optim.Adam(net.getAlphasPara(), lr=nas_initial_lr, weight_decay=weight_decay)
        return model_optimizer, nas_optimizer
    
def printNetWeight(net):
    for name, para in net.named_parameters():
        print(name, para)


    
def saveCheckPoint(kth, epoch, optimizer, net, lossRecord, accReocrd):
    print("save check point kth {} epoch {}".format(kth, epoch))
    if epoch==0:
        return 
    # print("net.state_dict()", net.state_dict())
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
        
        
def myTrain(kth, trainData, train_loader, val_loader, net, model_optimizer, nas_optimizer, criterion, writer):
    
    # calculate how many iterations
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
    alphaMonitor = AlphasMonitor()
    
    print("minibatch size: ", epoch_size)
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
            #! why this will be recreate per epoch, need to be fixed
            accRecord = {"train": record_train_acc, "val": record_val_acc, "test": record_test_acc}
            lossRecord = {"train": record_train_loss, "val": record_val_loss}
            
            #info handle alpha operation 
            if epoch in epoch_to_drop:
                # pass
                net.dropMinAlpha()
                model_optimizer, nas_optimizer = prepareOpt(net)
            if epoch >= cfg['start_train_nas_epoch']:
                # net.filtAlphas()
                net.normalizeAlphas()
                # net.normalizeByDivideSum()
                # net.saveMask(epoch, kth)
            
            train_batch_iterator = iter(train_loader)

        load_t0 = time.time()
        


        # load train data
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        train_images, train_labels = next(train_batch_iterator)
        val_batch_iterator = iter(val_loader)
        val_images, val_labels = next(val_batch_iterator)

        train_images = train_images.to(device)
        train_labels = train_labels.to(device)
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)


        model_optimizer.zero_grad(set_to_none=True)
        nas_optimizer.zero_grad(set_to_none=True)
        
        alphaMonitor.logAlphaDictPerIter(net, iteration)
        #info Forward pass
        train_outputs = net(train_images)
        #info calculate loss
        train_loss = criterion(train_outputs, train_labels)
        #info backward pass
        # print("weightCount ", weightCount(net))
        # print("gradCount ", gradCount(net))
        # val = input("Enter ")
        train_loss.backward()
        # val = input("Enter ")

        # print("increase mem", torch.cuda.memory_allocated(device=device)-mem)
        # print("weightCount ", weightCount(net))
        # print("gradCount ", gradCount(net))

        # print("epoch >= cfg['start_train_nas_epoch']", epoch >= cfg['start_train_nas_epoch'])
        # print("(epoch - cfg['start_train_nas_epoch']) % 2 == 0", (epoch - cfg['start_train_nas_epoch']) % 2 == 0)
        # print("epoch", epoch, "cfg['start_train_nas_epoch']", cfg['start_train_nas_epoch'])
        # alphaMonitor.logAlphasGradPerIteration(net, iteration)

        # train_transform_images = rotater(train_images)
        # train_transform_outputs = net(train_transform_images)
        # take turns to optimize weight and alphas
        #info update weight
        if epoch >= cfg['start_train_nas_epoch']:
            if (epoch - cfg['start_train_nas_epoch']) % 2 == 0:
                nas_optimizer.step()
            else:
                model_optimizer.step()
        else:
            model_optimizer.step()

            
        #info recording training process
        # model預測出來的結果 (訓練集)
        # record_train_loss.append(train_loss.item())
        # _, predictsTransform = torch.max(train_transform_outputs.data, 1)
        # coincident_output = (predicts == predictsTransform)
        #! Why she use validation directly at the end of an iteration.
        #! Usually we use validation after finishing all training.
        #! And chose the model generate with best accuracy on validation set
        #info do statistics after this epoch
        if (iteration % epoch_size == 0):
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


            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            
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
            #info record to tensorboard
            writer.add_scalar('Train_Loss/k='+str(kth), train_loss.item(), epoch)
            writer.add_scalar('Val_Loss/k='+str(kth), val_loss.item(), epoch)
            writer.add_scalar('train_Acc/k='+str(kth), trainAcc, epoch)
            writer.add_scalar('val_Acc/k='+str(kth), valAcc, epoch)
            writer.add_scalar('test_Acc/k='+str(kth), testAcc, epoch)
            accToTensorBoard = {
                "trainAcc": trainAcc,
                "valAcc": valAcc,
                "testAcc": testAcc,
            }
            writer.add_scalars('Acc/k='+str(kth), accToTensorBoard, epoch)
            last_epoch_val_acc = 100 * correct_images_val / total_images_val
            
            #info tensorboard to record wrong expectation
            # img_batch = np.zeros((16, 3, 100, 100))
            # for i in range(16):
            #     img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
            #     img_batch[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i
            # # print(train_images.size())
            # # print(img_batch)
            # for i in range(64):
            #     writer.add_image('my_image/{}'.format(i), train_images[i], 0)
            # exit(0)
            # print("coincident_output", coincident_output)
            # print(coincident_output.shape)
            # writer.add_images('my_image_batch', train_images[:, :, :, :], 0)
            # numOfWrongPredict = batch_size - coincident_output.sum()
            # print("numOfWrongPredict", numOfWrongPredict)
            # first = True
            # tmp = None
            # for i in range(len(coincident_output)):
                # if (coincident_output[i].data==False):
                #     if first==True:
                #         tmp = torch.cat((train_images[i], train_transform_images[i]), dim=0)
                #         first=False
                #     else:
                #         print("iteration", iteration)
                #         tmp = torch.cat((tmp, train_images[i], train_transform_images[i]), dim=0)
            # tmp = torch.rand((2*numOfWrongPredict, 3, 128, 128))
            # print("tmp.shape", tmp.shape)
            # tmp = torch.reshape(tmp, (2*numOfWrongPredict, 3, 128, 128))
            # print("reshpae tmp.shape", tmp.shape)
            # print("tmp[1][1]", tmp[1][1])
            
            # writer.add_images("helo", tmp, 0)
            # writer.add_images('epoch{}_iter{}'.format(epoch, iteration), tmp, 0)
            # tmp1 =  train_images[coincident_output, :, :, :]
            # print("tmp1.shape", tmp1.shape)
            
            # writer.add_images('my_image_batch1', tmp1, 0)
            #info save checkpoint model
            # accRecord = {"train": record_train_acc, "val": record_val_acc, "test": record_test_acc}
            # lossRecord = {"train": record_train_loss, "val": record_val_loss}
            # if epoch%3==0:
            #     pass
                # saveCheckPoint(kth, epoch, model_optimizer, net, lossRecord, accRecord)
        # transformImgTest.compare(net, train_images, predicts, train_labels, writer, iteration)
        # print("iteration", iteration)
        # net.getAlphasPara()
        
        # if iteration>=600:
        #     break


    lossRecord = {"train": record_train_loss, "val": record_val_loss}
    accRecord = {"train": record_train_acc, "val": record_val_acc, "test": record_test_acc}


    alphaMonitor.saveAllAlphas(kth)
    alphaMonitor.saveAllAlphasGrad(kth)
    torch.save(net.state_dict(), os.path.join(folder["savedCheckPoint"], cfg['name'] + str(kth) + '_Final.pt'))
    
    
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
        elif args.network == "nasmodel":
            cfg = cfg_nasmodel
        else:
            print('Model %s doesn\'t exist!' % (args.network))
            sys.exit(0)

        img_dim = cfg['image_size']
        num_gpu = cfg['ngpu']
        batch_size = cfg['batch_size']
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
        
        # transformImgTest = TransformImgTester(batch_size, kth=k)

        print("seed_img {}, seed_weight {} start at ".format(seed_img, seed_weight), getCurrentTime())
        print("training cfg", cfg)
        trainData, valData = prepareDataSet()
        trainDataLoader, valDataLoader = prepareDataLoader(trainData, valData)
        criterion = prepareLossFunction()
        net = prepareModel()

        model_optimizer, nas_optimizer = prepareOpt(net)
        last_epoch_val_ac, lossRecord, accRecord  = myTrain(k, trainData, trainDataLoader, valDataLoader, net, model_optimizer, nas_optimizer, criterion, writer)  # 進入model訓練
        #info record training processs
        alMonitor = AccLossMonitor(k, folder["pltSavedDir"], folder["accLossDir"], trainType="Nas")
        alMonitor.plotAccLineChart(accRecord)
        alMonitor.plotLossLineChart(lossRecord)
        alMonitor.saveAccLossNp(accRecord, lossRecord)
        valList.append(last_epoch_val_ac)
        print('train validate accuracy:', valList)
        
        datasetReviewer = DatasetReviewer(cfg["batch_size"],
                                        k,
                                        DatasetHandler.getOriginalDataset(trainDataSetFolder, cfg, seed_img), 
                                        device)
        datasetReviewer.makeSummary(trainDataLoader, writer, net)
        datasetReviewer.showReport()
        
        writer.close()
        if stdoutTofile:
            setStdoutToDefault(f)
        # exit() #* for examine why same initial value will get different trained model
    print('train validate accuracy:', valList)
        
        
        
        
        
        
        
        
        
        
        