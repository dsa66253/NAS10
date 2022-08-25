from os import listdir
from os.path import isfile, join
import numpy as np
#info handle .npy file name
#info delete time info from file name
#info retain kth and epoch informations



def getAllFileName(dirPath):
    return [f for f in listdir(dirPath) if isfile(join(dirPath, f))]



if __name__ =="__main__":
    dirPath = "./alpha_pdart_nodrop"
    fileNameList = getAllFileName(dirPath)
    print("number of files", len(fileNameList))
    
    for oldFileName in fileNameList:
        #info handle file name
        fileNameSplit = oldFileName.split("_")
        newFileName = "_".join(fileNameSplit[0:4])
        # print("newFilename", newFileName)
        
        #info load np array and save it with new fiel name
        oldFilePath = join(dirPath, oldFileName)
        # print("oldFilePath", oldFilePath)
        tmp = np.load(oldFilePath)
        newFilePath = join(dirPath, newFileName)
        # print("newFilePath", newFilePath)
        np.save(newFilePath, tmp)
        
    