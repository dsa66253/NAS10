from os import listdir
from os.path import isfile, join
import os
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
    sourceDir = "../dataset/train"
    desDir = "../dataset/test"

    sourceClassDirList = [x[0] for x in os.walk("../dataset/train")][1:]
    desClassDirList = [x[0] for x in os.walk("../dataset/test")][1:]
    for i in range(len(sourceClassDirList)):
        fileList = getAllFileName(sourceClassDirList[i])
        for j in range(140):
            # pass
            moveFile(sourceClassDirList[i], desClassDirList[i], fileList[j])
    