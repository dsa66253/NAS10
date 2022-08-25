import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
from data.config import PRIMITIVES, folder
from os import listdir
from data.config import cfg_nasmodel as cfg

numOfEpoch=cfg["epoch"]
# numOfEpoch = 25
kth=0
alphaFolder = "alpha_pdart_nodrop"
desFolder = "plot"
def loadAllAlphas(kth=0):
    listAlphas = []
    for epoch in range(numOfEpoch):
        tmp = np.load("./alpha_pdart_nodrop/alpha_prob_{}_{}.npy".format(kth, epoch))
        listAlphas.append(tmp)
    return listAlphas


#* create x axis labels 0, 5
#* labels also means build rectangle at which epoch is
def plot(kth, atLayer=0, atInnerCell=0):
    #info prepare needed data
    #* name:alphas_layer_innercell_op
    allAlphas = loadAllAlphas(kth)
    opList = ["alphas{}_{}_3".format(atLayer, atInnerCell),
            "alphas{}_{}_5".format(atLayer, atInnerCell),
            "alphas{}_{}_7".format(atLayer, atInnerCell),
            "alphas{}_{}_9".format(atLayer, atInnerCell),
            "alphas{}_{}_11".format(atLayer, atInnerCell),
            "alphas{}_{}_skip".format(atLayer, atInnerCell)
            ]
    opDic = {}
    for op in opList:
        opDic[op] = []
    for epoch in range(len(allAlphas)):
        for op in range(len(opList)):
            # print("handle epoch{} atinnercell{} op{}".format(epoch, atInnerCell, op))
            # print("allAlphas[epoch]", allAlphas)
            # print("allAlphas[epoch]", allAlphas[epoch])
            # print("allAlphas[epoch][atLayer]", allAlphas[epoch][atLayer])
            # print("allAlphas[epoch][atLayer][atInnerCell]", allAlphas[epoch][atLayer][atInnerCell])
            opDic[opList[op]].append( allAlphas[epoch][atLayer][atInnerCell][op] )

            
            
    #info put data into figure
    fig, ax = plt.subplots(figsize=(20,4))
    width = 0.13
    x = np.arange(len(allAlphas))  # the label locations
    for i in range(len(opList)):
        ax.bar(x + i*width, opDic[opList[i]], width, label=opList[i], align='edge')
    
    
    ax.set_ylabel('alphas probability')
    ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel('epoch')
    ax.set_xticks(range(0, numOfEpoch, 1))
    ax.set_title('layer{} innercell{}'.format(atLayer, atInnerCell))
    # ax.set_xticks(x, labels)
    ax.legend(loc=2)
    # ax.grid(True);
    plt.savefig('./plot/' + '{}th_layer{}_innercell{}'.format(kth, atLayer, atInnerCell) + '.png')


    # plt.savefig('./weights_pdarts_nodrop/' + 'layer' + str(currentLayer) + '.png')

def loadAlphasAtEpoch(kth, epoch):
    return  np.load("./alpha_pdart_nodrop/alpha_prob_{}_{}.npy".format(str(kth), str(epoch)))
    
def drawBar():
    allAlphas = loadAllAlphas(kth)

    numOfEpoch = len(allAlphas)
    numOfLayer = len(allAlphas[0])
    numOfInnerCell = len(allAlphas[0][0])
    numOfOp = len(allAlphas[0][0][0])

    print("alphas architecture:numOfEpoch {}, numOfLayer {}, numOfInnerCell {}, numOfOp {}".format(numOfEpoch, numOfLayer, numOfInnerCell, numOfOp))
    for kth in range(3):
        for layer in range(numOfLayer):
            for innerCell in range(numOfInnerCell):
                plot(kth, layer, atInnerCell=innerCell)
def plot_line_chart_layer_innercell(kth, atLayer=0, atInnerCell=0):
    #info prepare needed data
    print("draw {}th layer{} innercell{}".format(kth, atLayer, atInnerCell))
    #* name:alphas_layer_innercell_op
    allAlphas = np.load("./alpha_pdart_nodrop/allAlphas_{}.npy".format(kth))
    opList = []
    for i in range(len(PRIMITIVES)):
        opList.append(PRIMITIVES[i]+"_{}_{}".format(atLayer, atInnerCell))
    opDic = {}
    for op in opList:
        opDic[op] = []
    for iteraion in range(len(allAlphas)):
        for op in range(len(opList)):
            opDic[opList[op]].append( allAlphas[iteraion][atLayer][atInnerCell][op] )
            
            
    #info put data into figure
    numOfIteration = len(allAlphas)
    numOfRow = 3
    numOfCol = 1
    fig, axs = plt.subplots(numOfRow, numOfCol, figsize=(20, 6), constrained_layout=True)
    x = np.arange(len(allAlphas))  # the label locations
    for i in range(len(opList)):
        # ax.bar(x + i*width, opDic[opList[i]], width, label=opList[i], align='edge')
        splits = np.array_split(opDic[opList[i]], numOfRow)
        for row in range(len(splits)):
            axs[row].plot(splits[row], label=opList[i])
            
            axs[row].set_ylabel('alphas probability')
            # axs[row].set_yticks([0.1, 0.2, 0.3, 0.4])
            # axs[row].set_yticks(np.arange(0, 0.01, 0.001))
            axs[row].set_title('layer{} innercell{}'.format(atLayer, atInnerCell))
            
            #info data need to transfer iteration to epoch
            baseTick = row * numOfIteration//len(splits)
            numOfTick = numOfIteration//len(splits)
            iterPerEpoch = numOfIteration//numOfEpoch
            axs[row].set_xticks( range( 0, numOfTick, iterPerEpoch) )
            axs[row].set_xticklabels( np.array( range( baseTick, baseTick+numOfTick, iterPerEpoch) )//iterPerEpoch )
            axs[row].set_xlabel('epoch')
            axs[row].legend(loc=2)
    # plt.show()
    plt.savefig('./plot/' + 'lineChart_{}th_layer{}_innercell{}'.format(kth, atLayer, atInnerCell) + '.png')
def plot_line_chart_all(kth):
    
    allAlphas = np.load("./alpha_pdart_nodrop/allAlphas_{}.npy".format(kth))
    numOfIteration = len(allAlphas)
    numOfLayer = len(allAlphas[0])
    numOfInnerCell = len(allAlphas[0][0])
    numOfOp = len(allAlphas[0][0][0])
    print("alphas architecture:numOfIteration {}, numOfLayer {}, numOfInnerCell {}, numOfOp {}".format(numOfIteration, numOfLayer, numOfInnerCell, numOfOp))
    for layer in range(numOfLayer):
        for innerCell in range(numOfInnerCell):
            plot_line_chart_layer_innercell(kth, layer, atInnerCell=innerCell)
def plot_line_chart_all_file():
    fileNameList = []

    for fileName in sorted(listdir(folder["alpha_pdart_nodrop"])):
        #* load alpha npy file
        fileNameList.append(os.path.join(folder["alpha_pdart_nodrop"], fileName))
        filePath = os.path.join(folder["alpha_pdart_nodrop"], fileName)
        allAlphas = np.load(filePath)
        plot_line_chart_innercell(allAlphas, fileName=fileName.split(".")[0])
        # allAlphas = [[0.2, 0.2, 0.2, 0.2, 0.2]
        #         ,[0.2, 0.2, 0.2, 0.2, 0.2]]

        # plot_line_char_innercell(allAlphas, fileName)
        # break

    # allAlphas = [[0.5, 0.2, 0.2, 0.3, 0.1]
    #             ,[0.2, 0.2, 0.2, 0.2, 0.2]]
    # tmp = np.array(allAlphas)
    # for i in range(100):
    #     tmp = np.append(tmp, allAlphas, axis=0)
    # allAlphas = np.array(tmp)

    # plot_line_char_innercell(allAlphas, fileName="tmp")

    

def plot_line_chart_innercell(allAlphas, fileName=""):
    alphaDict = {}
    for key in PRIMITIVES:
        alphaDict[key] = []
    for index in range(len(PRIMITIVES)):
        for iteration in range(len(allAlphas)):
            alphaDict[PRIMITIVES[index]].append( allAlphas[iteration][index] )
    # allAlphas = [[0.2 0.2 0.2 0.2 0.2]
    #         ,[0.2 0.2 0.2 0.2 0.2]]
    numOfIteration = len(allAlphas)
    numOfRow = 3
    numOfCol = 1
    fig, axs = plt.subplots(numOfRow, numOfCol, figsize=(20, 6), constrained_layout=True)
    x = np.arange(len(allAlphas))  # the label locations
    totalTickRow = numOfIteration // numOfRow
    totalEpochRow = numOfEpoch // numOfRow
    iterPerEpoch = numOfIteration//numOfEpoch
    for i in range(len(PRIMITIVES)):
        #* print each conv line
        # splits = np.array_split(alphaDict[PRIMITIVES[i]], numOfRow)
        # #! split didn't match tick labels
        # axs[0].plot(alphaDict[PRIMITIVES[i]], label=PRIMITIVES[i])
        for row in range(numOfRow):

            splits = alphaDict[PRIMITIVES[i]][row*totalEpochRow*iterPerEpoch:(row+1)*totalEpochRow*iterPerEpoch]
            axs[row].plot(splits, label=PRIMITIVES[i])
            
            axs[row].set_ylabel('alphas value')
            # axs[row].set_yticks([0.1, 0.2, 0.3, 0.4])
            # axs[row].set_yticks(np.arange(0, 0.01, 0.001))
            axs[row].set_title(fileName)
            
            #info data need to transfer iteration to epoch

            # print("numOfIteration", numOfIteration)
            # print("numOfEpoch", numOfEpoch)
            # print("iterPerEpoch", iterPerEpoch)
            baseTick = row * numOfIteration//numOfRow
            axs[row].set_xticks( range( 0, totalTickRow, iterPerEpoch) )
            axs[row].set_xticklabels( np.array( range( baseTick, baseTick+totalTickRow, iterPerEpoch) )//iterPerEpoch )
            axs[row].set_xticklabels( np.array( range( baseTick, baseTick+totalTickRow, iterPerEpoch) )//iterPerEpoch )
            axs[row].set_xlabel('epoch')
            axs[row].legend(loc=2)
    # plt.show()
    print("save to ", os.path.join(desFolder, fileName)+ '.png')
    plt.savefig(os.path.join(desFolder, fileName)+ '.png')


if __name__=="__main__":
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    plot_line_chart_all_file()
    print()
    # for kth in range(3):
    #     plot_line_chart_all(kth)


    