from data.config import cfg_nasmodel as cfg
import torchvision.transforms as T
import numpy as np
import torch
from datasetPractice import DatasetHandler
from data.config import trainDataSetFolder
from torchvision import datasets
class DatasetReviewer():
    def __init__(self, batch_size, kth, allData, device):
        self.device = device
        self.numOfClasses = cfg["numOfClasses"]
        self.batch_size = batch_size
        self.rotater = T.RandomRotation(degrees=10)
        self.flipTransform = T.RandomHorizontalFlip(p=1)
        self.kth = kth
        self.numOfClasses = cfg["numOfClasses"]
        # print(type(allData))
        allData = datasets.ImageFolder(trainDataSetFolder)
        self.idx_to_class = [name for name in allData.class_to_idx]
        
        self.static = {}
        for nameOfClass in allData.class_to_idx:
            self.static[nameOfClass] = DataStatic(nameOfClass, allData.class_to_idx[nameOfClass])
        # print("self.static", self.static)
        # print("self.static[Millipede]", self.static["Millipede"])
    def makeSummary(self, dataLoader, writer, net):
        dataLoaderIter = iter(dataLoader)
        batch = 0
        net.eval()
        for train_images, train_labels in dataLoaderIter:
            train_images = train_images.to(self.device)
            train_labels = train_labels.to(self.device)
            train_outputs = net(train_images)
            _, predicts = torch.max(train_outputs.data, 1)
            
            #info transform data
            train_transform_images = self.flipTransform(train_images)
            train_transform_outputs = net(train_transform_images)
            _, predictsTransform = torch.max(train_transform_outputs.data, 1)
            
            predictsTransformTF = (train_labels == predictsTransform)
            predictsTF = (train_labels == predicts)
            
            # print("train_labels", train_labels)
            for i in range(len(train_labels)):
                self.static[self.idx_to_class[train_labels[i]]].classCount += 1
                if (predictsTF[i].data==True and predictsTransformTF[i].data==True):
                    self.static[self.idx_to_class[train_labels[i]]].ttCount += 1
                elif((predictsTF[i].data==True and predictsTransformTF[i].data==False)):
                    self.static[self.idx_to_class[train_labels[i]]].tfCount += 1
                elif((predictsTF[i].data==False and predictsTransformTF[i].data==True)):
                    self.static[self.idx_to_class[train_labels[i]]].ftCount += 1
                elif((predictsTF[i].data==False and predictsTransformTF[i].data==False)):
                    self.static[self.idx_to_class[train_labels[i]]].ffCount += 1
                # self.static[self.idx_to_class[label]].totalcount += 1
            
            #info save img to tensorbaord 
            # writer.add_images('my_image_batch', train_images[:, :, :, :], 0)
            numOfWrongPredict = 0
            # print("numOfWrongPredict", numOfWrongPredict)
            
            for i in range(len(train_labels)):
                tmp = None
                first = True
                # print("comapre ", train_labels[i], predicts[i], predictsTransform[i])
                if (predictsTF[i].data==True and predictsTransformTF[i].data==False):
                    numOfWrongPredict = numOfWrongPredict + 1
                    # print("self.static[self.idx_to_class[train_labels[i]]]", type(self.static[self.idx_to_class[train_labels[i]]]))
                    # print("self.static[self.idx_to_class[train_labels[i]]][imgs]", self.static[self.idx_to_class[train_labels[i]]].imgs)
                    if (self.static[self.idx_to_class[train_labels[i]]].imgs==None):
                        tmp = torch.cat((train_images[i], train_transform_images[i]), dim=0)
                        self.static[self.idx_to_class[train_labels[i]]].imgs = tmp
                    else:
                        tmp = torch.cat((train_images[i], train_transform_images[i]), dim=0)
                        self.static[self.idx_to_class[train_labels[i]]].imgs = torch.cat((self.static[self.idx_to_class[train_labels[i]]].imgs, tmp), dim=0)
            batch += 1
            # if batch >=10:
            #     break
        for nameOfClass in self.static:
            if (self.static[nameOfClass].imgs != None):
                tmp = torch.reshape(self.static[nameOfClass].imgs, (2*self.static[nameOfClass].tfCount, 3, 128, 128))
                writer.add_images("{}th/tf_{}_{}".format(self.kth, nameOfClass, self.static[nameOfClass].calAcc(0)), tmp, 0)
        net.train()
    
    def showReport(self):
        for key in self.static:
            print(self.static[key])
    
class DataStatic():
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.classCount = 0
        self.imgs = None
        self.ttCount = 0
        self.tfCount = 0
        self.ffCount = 0
        self.ftCount = 0
        
    def calAcc(self, whichTFCondi):
        if whichTFCondi==0:
            return round(self.ttCount/self.classCount*100)
        elif whichTFCondi==1:
            return  round(self.tfCount/self.classCount*100)
        elif whichTFCondi==2:
            return round(self.ffCount/self.classCount*100)
        elif whichTFCondi==3:
            return round(self.ftCount/self.classCount*100)
    def __str__(self):
        
        # return "helo"
        return "name:{}, index:{}, classCount:{}, ttCount:{}, tfCount:{}, ftCount{}, ffCount{}"\
        .format(self.name, self.index, self.classCount, self.ttCount, self.tfCount, self.ftCount, self.ffCount)
        
        
if __name__ == "__main__":
    k=0
    trainDataSetFolder = "../datasetPractice/train"
    seed_img=20
    datasetReviewer = DatasetReviewer(cfg["batch_size"],
                                        k,
                                        DatasetHandler.getOriginalDataset(trainDataSetFolder, cfg, seed_img), 
                                        "cuda")
    pass