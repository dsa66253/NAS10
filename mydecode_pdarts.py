import os
import json
# from re import S
from os import listdir
import numpy as np
from feature.make_dir import makeDir
import sys
from data.config import featureMap, PRIMITIVES, folder
# this file just use to plot figure that shows alphas' variation during training
def setStdoutToFile(filePath):
    print("std output to ", filePath)
    f = open(filePath, 'w')
    sys.stdout = f
    return f

def setStdoutToDefault(f):
    f.close()
    sys.stdout = sys.__stdout__
def tensor_to_list(input_op):
    return input_op.numpy().tolist()
def pickSecondMax(input):
    # set the max to zero to and find the max again 
    # to get the index of second large value of original input
    alphas = np.copy(input)
    alphasMaxIndex = np.argmax(alphas, -1)

    for i in range(len(alphas)):
        alphas[i, alphasMaxIndex[i]] = 0
    alphasSecondMaxIndex = np.argmax(alphas, -1)

    return alphasSecondMaxIndex

def loadAllAlphas():
    listAlphas = []
    epoch = 45
    for epoch in range(epoch):
        tmp = np.load("./alpha_pdart_nodrop/alpha_prob_0_{}.npy".format(epoch))
        listAlphas.append(tmp)
    return listAlphas

def loadAlphasAtEpoch(kth, epoch):
    return  np.load("./alpha_pdart_nodrop/alpha_prob_{}_{}.npy".format(str(kth), str(epoch)))
def decodeAlphas(kth):
    #* get index of innercell which hase greatest alphas value
    genotype_filename = os.path.join('./weights_pdarts_nodrop/',
                        'genotype_' + str(kth))
    lastEpoch = 44
    lastAlphas = loadAlphasAtEpoch(kth, lastEpoch)
    maxAlphasIndex = np.argmax(lastAlphas, axis=-1)
    
    # choose top two alphas
    # print("lastAlphas", lastAlphas)
    indexOfSortAlphas = np.argsort(lastAlphas, axis=-1)
    # twoLargestIndex = indexOfSortAlphas[:, 1:, 5:]
    oneLargestIndex = indexOfSortAlphas[:, :, -1]
    # twoLargestIndex = np.reshape([0, 0, 0, 0, 0], twoLargestIndex.shape)
    # oneLargestIndex = np.reshape([4, 1, 0, 0, 0], oneLargestIndex.shape)
    # print(oneLargestIndex.shape)
    
    # print("indexOfSortAlphas", indexOfSortAlphas)
    
    # print("twoLargestIndex", twoLargestIndex)
    oneLargestIndex = np.reshape(oneLargestIndex, (5, 1, 1))
    # oneLargestIndex = np.reshape([4, 1, 0, 0, 0], (5, 1, 1))
    # twoLargestIndex = np.reshape(twoLargestIndex, (5, 2))
    # print("oneLargestIndex", oneLargestIndex.shape)
    np.save(genotype_filename, oneLargestIndex)
    
    # print("finish decode and save genotype:", maxAlphasIndex)
    return oneLargestIndex
    
def manualAssign(kth):

    makeDir("./weights_pdarts_nodrop/")
    genotype_filename = os.path.join('./weights_pdarts_nodrop/',
                    'genotype_' + str(kth))
        
    arch = np.reshape([4, 1, 0, 0, 0], (5, 1, 1))
    np.save(genotype_filename, arch)
    return arch
def decodeInnerCell(allAlphas):
    takeNumOfOp = 1
    finalAlpha = allAlphas[-1] #* take the last epoch

    sortAlphaIndex = np.argsort(finalAlpha) #* from least to largest
    sortAlphaIndex = sortAlphaIndex[::-1] #* reverse ndarray
    res = np.full_like(finalAlpha, 0, dtype=np.int32)
    for i in range(takeNumOfOp):
        res[sortAlphaIndex[i]] = 1
    return res.tolist() #* make ndarray to list
def decodeAllInnerCell(kth):
    fileNameList = []
    decodeDict = {}
    #* split file according to different kth
    for fileName in sorted(listdir(folder["alpha_pdart_nodrop"])):
        
        if fileName.split("th")[0]==str(kth):
            fileNameList.append(fileName)
    for fileName in fileNameList:
        #* load alpha npy file
        filePath = os.path.join(folder["alpha_pdart_nodrop"], fileName)
        allAlphas = np.load(filePath)
        key = fileName.split(".")[0]
        key = key.split("th_")[1]
        decodeDict[key] = decodeInnerCell(allAlphas)
        
    print(json.dumps(decodeDict, indent=4)) #* make ndarray to list
    return decodeDict

if __name__ == '__main__': 
    # filePath = "./decode/decode.json"
    # setStdoutToFile(filePath)
    # decodeAllInnerCell()
    # f = open("./decode/0th_decode.json")
    # # returns JSON object as 
    # # a dictionary
    # data = json.load(f)
    # print(data)
    # for key in data:
    #     print(key, data[key])
    # exit()
    for kth in range(3):
        filePath = "./decode/{}th_decode.json".format(kth)
        f = setStdoutToFile(filePath)
        
        decodeAllInnerCell(kth)
        
        setStdoutToDefault(f)
        # break
    # for kth in range(3):
        # print(decodeAlphas(kth))
        # print(manualAssign(kth))

    exit()
