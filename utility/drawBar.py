import matplotlib.pyplot as plt
import torch
import numpy as np
torch.manual_seed(0)
# alphas = (torch.rand(5, 2, 5)*10).numpy()
# for epoch in range(45):
#     torch.manual_seed(epoch)
#     alphas = (torch.rand(5, 2, 5)*10).numpy()
#     np.save("./alphas/alphas_epoch{}".format(epoch), alphas)

import os
import matplotlib.pyplot as plt
import numpy as np

def loadAllAlphas():
    listAlphas = []
    for epoch in range(numOfEpoch):
        tmp = np.load("./alpha_pdart_nodrop/alpha_prob_0_{}.npy".format(epoch))
        listAlphas.append(tmp)
    return listAlphas


#* create x axis labels 0, 5
#* labels also means build rectangle at which epoch is
def plot(atLayer=0, atInnerCell=0):
    
    # the width of the bars
    fig, ax = plt.subplots(figsize=(16,4))
    
    #info prepare needed data
    #* name:alphas_layer_innercell_op
    allAlphas = loadAllAlphas()
    opList = ["alphas{}_{}_3".format(atLayer, atInnerCell),
            "alphas{}_{}_5".format(atLayer, atInnerCell),
            "alphas{}_{}_7".format(atLayer, atInnerCell),
            "alphas{}_{}_9".format(atLayer, atInnerCell),
            "alphas{}_{}_11".format(atLayer, atInnerCell)]
    opDic = {}
    for op in opList:
        opDic[op] = []
    for alphaEpoch in allAlphas:
        atOp = 0
        for op in opList:
            opDic[op].append( alphaEpoch[atLayer][atInnerCell][atOp] )
            atOp = atOp + 1
            
    #info put data into figure
    width = 0.15
    x = np.arange(len(allAlphas))  # the label locations
    for i in range(len(opList)):
        ax.bar(x + i*width, opDic[opList[i]], width, label=opList[i], align='edge')
        
    ax.set_ylabel('alphas probability')
    ax.set_yticks([0.1, 0.4])
    ax.set_xlabel('epoch')
    ax.set_xticks(range(0, numOfEpoch, 5))
    ax.set_title('layer{} innercell{}'.format(atLayer, atInnerCell))
    # ax.set_xticks(x, labels)
    ax.legend(loc=2)
    # ax.grid(True);
    plt.savefig('./plot/' + 'layer{}_innercell{}'.format(atLayer, atInnerCell) + '.png')


    # plt.savefig('./weights_pdarts_nodrop/' + 'layer' + str(currentLayer) + '.png')

def loadAlphasAtEpoch(kth, epoch):
    return  np.load("./alpha_pdart_nodrop/alpha_prob_{}_{}.npy".format(str(kth), str(epoch)))
def loadAlphaEpoch(epoch):
    dirPath = "alpha_pdart_nodrop"
    filePath = os.path.join("alpha_prob_{}_{}.npy")
    np.load()
    
def test():
    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    men_means = [20, 34, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, men_means, width, label='Men')
    rects2 = ax.bar(x + width/2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    # ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

# fig.tight_layout()
if __name__=="__main__":
    numOfLayer = 5
    numOfInnerCell=1
    for layer in range(numOfLayer):
        for innerCell in range(numOfInnerCell):
            plot(layer, innerCell)
    # plot(44)

    # print(np.random.randn(2, 10))
    # print(data1)
    exit()
    print(loadAlphasAtEpoch(0, 0))
    fig = plt.figure()  # an empty figure with no Axes
    plot.show()
    fig, ax = plt.subplots()  # a figure with a single Axes
    fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
    




exit()
def loadAlphasAtEpoch(epoch):
    return  np.load("./alpha_pdart_nodrop/alpha_prob_0_{}.npy".format(str(epoch)))

genotype_filename = os.path.join('./weights_pdarts_nodrop/',
                        'genotype_' + str(0))
lastAlphas = loadAlphasAtEpoch(44)
maxAlphasIndex = np.argmax(lastAlphas, -1)
# np.save(genotype_filename, maxAlphasIndex)

# print(lastAlphas, maxAlphasIndex)
# print(maxAlphasIndex)
allAlphas = loadAllAlphas()
for i in range(len(allAlphas)):
    break
    print(i, allAlphas[i])

exit()

# print(len(listAlphas))
fig, ax = plt.subplots()


# print(len(alphas))
alphas0 = listAlphas[0]
width=0.1
numOfEpoch = len(listAlphas)
numOfInnerCell = len(alphas0[0])
numOfOpPerInnerCell = len(alphas0[0][0])
xAxisBase = np.arange(numOfEpoch)
numOfLayer = len(listAlphas[0])
for epoch in range(numOfEpoch):
    tranformedAlphas= []
    tranformedAlphas = listAlphas[epoch]
for i in range(10):
    print(np.load("./alpha_pdart_nodrop/alpha_prob_0_{}.npy".format(i)))
# exit()
for epoch in range(numOfEpoch):
    
    for layer in numOfLayer:
        for op in []:
            ax.bar(xAxisBase+2*width, alphas[layer][0])