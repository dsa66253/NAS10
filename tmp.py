from os import listdir
from os.path import isfile, join
import os
import torch
from  models.mymodel import InnerCell, Layer, Model

'''
pick up 140 image from training set to test set
'''
def getAllFileName(dirPath):
    return [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)
def moveFile(sourceDir, desDir, fileName):
    sourcePath = os.path.join(sourceDir, fileName)
    desPath = os.path.join(desDir, fileName)
    os.rename(sourcePath, desPath)
if __name__=="__main__":
    print(torch.cuda.is_available())

    torch.manual_seed(10)
    input = torch.rand(3, 3, 128, 128)
    net = Model()
    output = net(input)
    output = output.sum()
    output.backward()
    # output = net(input)
    # print(.shape)
    exit()



    torch.manual_seed(10)
    input = torch.rand(3, 3, 64, 64)
    net = Layer(1, 0, 3, 96, 1, 1, [0, 1],  "testLayer")
    output = net(input)
    output = output.sum()
    output.backward()
    output = net(input)
    # print(.shape)
    exit()





    torch.manual_seed(10)
    input = torch.rand(3, 3, 64, 64)
    innercell = InnerCell(3, 96, 1, None, "testLayer")
    output = innercell(input)
    output = output.sum()
    output.backward()
    output = innercell(input)
    # print(.shape)
    exit()



    sourceDir = "../dataset/train"
    desDir = "../dataset/test"

    sourceClassDirList = [x[0] for x in os.walk("../dataset/train")][1:]
    desClassDirList = [x[0] for x in os.walk("../dataset/test")][1:]
    for i in range(len(sourceClassDirList)):
        fileList = getAllFileName(sourceClassDirList[i])
        for j in range(140):
            # pass
            moveFile(sourceClassDirList[i], desClassDirList[i], fileList[j])
    