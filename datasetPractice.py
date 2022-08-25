from __future__ import print_function
from torchvision import transforms
import torch
import torch.optim as optim
# import torch.backends.cudnn as cudnn
import argparse
from torch import nn
from torchvision import datasets
from data.config import cfg_newnasmodel as cfg
from tensorboardX import SummaryWriter
import numpy as np
from data.config import folder
from feature.normalize import normalize
from feature.make_dir import makeDir
# from feature.split_data import split_data
from feature.random_seed import set_seed_cpu
from PIL import ImageFile
from tqdm import tqdm
from retrainModel import NewNasModel
# from alexnet import Baseline
from feature.utility import plot_acc_curve, setStdoutToFile, setStdoutToDefault
from feature.utility import getCurrentTime, accelerateByGpuAlgo, get_device, plot_loss_curve
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
trainDataSetFolder = "../datasetPractice/train"

class DatasetHandler():
    def __init__(self, trainDataSetFolder, cfg, seed=10):
        self.seed = seed
        self.augmentDatasetList = []
        self.trainDataSetFolder = trainDataSetFolder
        self.normalize = self.resize(cfg["image_size"])
        self.originalData = datasets.ImageFolder(self.trainDataSetFolder, transform=transforms.Compose([
                self.normalize,
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
        self.originalTrainDataset, self.originalValDataset = self.split_data(self.originalData, 0.2)
        self.trainDataset = self.originalTrainDataset
        self.augmentDatasetList.append(self.originalTrainDataset)
    def split_data(self, all_data, ratio=0.2):
        n = len(all_data)  # total number of examples
        n_val = int(ratio * n)  # take ~10% for val
        set_seed_cpu(self.seed)
        train_data, val_data = torch.utils.data.random_split(all_data, [(n - n_val), n_val])
        return train_data, val_data
    def resize(self, img_dim):
        return transforms.Compose([transforms.Resize(img_dim),
                                                transforms.CenterCrop(img_dim),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
    def addAugmentDataset(self, augmentF):
        # newAugmentData = augmentF(self.originalTrainData)
        
        try:
            newAugmentDataset = datasets.ImageFolder(self.trainDataSetFolder, transform=transforms.Compose([
                self.normalize,
                augmentF
            ]))
            newAugmentTrainDataset, _ = self.split_data(newAugmentDataset, 0.2)
        except Exception as e:
            print("Fail to load data set from: ",  trainDataSetFolder)
            print(e)
            exit()
        # self.augmentDatasetList.append(newAugmentData)
        print("self.trainDataset", type(self.trainDataset))
        print("newAugmentTrainDataset", type(newAugmentTrainDataset))
        self.trainDataset = torch.utils.data.ConcatDataset([self.trainDataset, newAugmentTrainDataset])
        
    def getTrainDataset(self):
        return self.trainDataset
    
    def getValDataset(self):
        return self.originalValDataset
    
    @staticmethod
    def getOriginalDataset(trainDataSetFolder, cfg, seed=10):
        datsetHandle = DatasetHandler(trainDataSetFolder, cfg, seed)
        return datsetHandle.originalTrainDataset
    def getLength(self):
        return len(self.trainDataset)
    def __getitem__(self, index):
        return self.trainDataset[index]
    # def prepareDataSet(self, augmentF): 
    #     #info prepare dataset
    #     # print('Loading Dataset from {} with seed_img {}'.format(trainDataSetFolder, str(self.seed)))
    #     train_transforms = self.normalize(10, cfg['image_size'])  # 正規化照片
        
    #     try:
    #         all_data = datasets.ImageFolder(trainDataSetFolder, transform=transforms.Compose([
                
    #         ]))
    #         # print(all_data.find_classes())
            
    #         print("all_data.class_to_idx", all_data.class_to_idx)
    #         # exit()
    #     except Exception as e:
    #         print("Fail to load data set from: ",  trainDataSetFolder)
    #         print(e)
    #         exit()
    #     train_data, val_data = split_data(all_data, 0.2)  # 切訓練集跟驗證集
    #     return all_data
def printImage(train_data, index):

    fig, axes = plt.subplots(len(train_data)//5, 5)
    for i in range(len(train_data)):
        img, label = train_data[i]
        # print(i//5, i%5)

        if len(train_data)//5==1:
            axes[i%5].imshow(img.permute(1, 2, 0))
        else:
            axes[i//5, i%5].imshow(img.permute(1, 2, 0))
        # if i%5==4:
    plt.savefig('foo{}.png'.format(index))   
    

    
if __name__ == "__main__":
    datasetHandler = DatasetHandler(trainDataSetFolder, cfg, 10)
    print(datasetHandler.getLength())
    printImage(datasetHandler.getTrainDataset(), "0")

    datasetHandler.addAugmentDataset(transforms.RandomHorizontalFlip(p=1))
    print(datasetHandler.getLength())
    printImage(datasetHandler.getTrainDataset(), "1")
    
    datasetHandler.addAugmentDataset(transforms.RandomGrayscale(p=1))
    print(datasetHandler.getLength())
    printImage(datasetHandler.getTrainDataset(), "2")
    
    # train_data = datasetHandler.getTrainDataset()


    
    
    exit()
    set_seed_cpu(20)
    train_data, val_data = prepareDataSet()
    set_seed_cpu(20)
    train_data2, val_data2 = prepareDataSet()
    # print(len(train_data), len(val_data))
    # figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    fig, axes = plt.subplots(2, 5)
    
    
    
    
    print(len(train_data))
    for i in range(len(train_data)+len(train_data2)):
        if i//5==0:
            img, label = train_data[i-1]
            print(i//5, i%5)
            axes[i//5, i%5].imshow(img.permute(1, 2, 0))
        else:
            img, label = train_data2[(i-1)%5]
            print(i//5, i%5)
            axes[i//5, i%5].imshow(img.permute(1, 2, 0))
        # print(type(axes))
        # axes[0][i].add_image(img.numpy())
        # plt.axis("off")
        # plt.savefig('foo.png')
        # plt.imshow(img.permute(1, 2, 0), cmap="gray")
    # plt.show()
    plt.savefig('foo.png')
        