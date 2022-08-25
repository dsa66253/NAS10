import os
from os import listdir
from os.path import isfile, join
import argparse
import pathlib
def parseArgs():
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument("--logDir", default="1", help="echo the string you use here")
    args = parser.parse_args()
    return args

def getAllFileName(dirPath):
    return [f for f in listdir(dirPath) if isfile(join(dirPath, f))]

def makeDir(folderPath):
    i = 1
    while True:
        # print(folderPath)
        if os.path.exists(folderPath):
            folderPath = os.path.join(os.path.split(folderPath)[0], str(i))
        else:
            os.makedirs(folderPath)
            break
        i = i + 1
    return folderPath
        
def moveLog(fileNameList, sourceDir, desDir):
    print("moving log from {} to {}".format(sourceDir, desDir))
    
    for fileName  in fileNameList:
        sourcePath = os.path.join(sourceDir, fileName)
        desPath = os.path.join(desDir, fileName)
        os.rename(sourcePath, desPath)
        
def moveAlphas():
    toDir = "1"
    sourceAlphasDir = "./alpha_pdart_nodrop"
    sourceLogDir = "./log"
    desLogDir = "./log"
    args = parseArgs()
    toDir = args.logDir
    
    desDir =os.path.join(desLogDir, toDir)
    desDir = makeDir(desDir)
    #info move alphas
    # print("move alphas from {} to {}".format(sourceAlphasDir, desAlphasDir))
    fileNameList = getAllFileName(sourceAlphasDir)
    moveLog(fileNameList, sourceAlphasDir,  desDir)
        
if __name__ =="__main__":
    toDir = "1"
    sourceAlphasDir = "./alpha_pdart_nodrop"
    sourceLogDir = "./log"
    desLogDir = "./log"
    args = parseArgs()
    toDir = args.logDir
    
    desDir =os.path.join(desLogDir, toDir)
    desDir = makeDir(desDir)
    #info move alphas
    # print("move alphas from {} to {}".format(sourceAlphasDir, desAlphasDir))
    fileNameList = getAllFileName(sourceAlphasDir)
    moveLog(fileNameList, sourceAlphasDir,  desDir)
    
    #info move log
    # print("move alphas from {} to {}".format(sourceLogDir, desLogDir))
    fileNameList = getAllFileName(sourceLogDir)
    moveLog(fileNameList, sourceLogDir,  desDir)
    
    