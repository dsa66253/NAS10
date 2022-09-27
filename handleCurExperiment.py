import sys, os, json
from os import listdir
from os.path import isfile, join
import shutil
def makeDir(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
def setStdoutToFile(filePath):
    f = open(filePath, 'w')
    sys.stdout = f
    return f
def getCurExpName():
    # load json
    filePath = os.path.join("./experiments.json")
    f = open(filePath)
    exp = json.load(f)
    finish = False
    curExpName = None
    for expName in exp:
        if exp[expName]==0:
            print(json.dumps({expName:1}, indent=4)) #* make ndarray to list
            exp[expName]=1
            curExpName = expName
            break
            
    setStdoutToFile("./experiments.json")
    print(json.dumps(exp, indent=4)) #* make ndarray to list
    return curExpName
def moveFilesToLog():
    pass
def getAllFileName(dirPath):
    return [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
def moveLog(fileNameList, sourceDir, desDir):
    # print("moving log from {} to {}".format(sourceDir, desDir))
    
    for fileName  in fileNameList:
        sourcePath = os.path.join(sourceDir, fileName)
        desPath = os.path.join(desDir, fileName)
        os.rename(sourcePath, desPath)
def copyFile(fileNameList, sourceDir, desDir):
    for fileName  in fileNameList:
        sourcePath = os.path.join(sourceDir, fileName)
        desPath = os.path.join(desDir, fileName)
        shutil.copyfile(sourcePath, desPath)
def openCurExp():
    filePath = os.path.join("./curExperiment.json")
    f = open(filePath)
    exp = json.load(f)
    for key in exp:
        return key 
if __name__=="__main__":
    curExpName = openCurExp()
    desDir = join("./log", curExpName)
    # copy alpha
    fileNameList = getAllFileName("./alpha_pdart_nodrop")
    desDir2 =  join(desDir, "alpha_pdart_nodrop")
    makeDir(desDir2)
    copyFile(fileNameList, "./alpha_pdart_nodrop", desDir2)
    # copy plot
    fileNameList = getAllFileName("./plot")
    desDir2 =  join(desDir, "plot")
    makeDir(desDir2)
    copyFile(fileNameList, "./plot", desDir2)
    # copy accloss
    fileNameList = getAllFileName("./accLoss")
    desDir2 =  join(desDir, "accLoss")
    makeDir(desDir2)
    copyFile(fileNameList, "./accLoss", desDir2)
    # copy log
    fileNameList = getAllFileName("./log")
    desDir2 =  join(desDir, "log")
    makeDir(desDir2)
    copyFile(fileNameList, "./log", desDir2)




