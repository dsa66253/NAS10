import subprocess
import json, os, sys, copy
from os.path import isfile, join
from data.config import folder, cfg_newnasmodel as cfg
def makeAllDir():
    for folderName in folder:
        print("making folder ", folder[folderName])
        makeDir(folder[folderName])
def makeDir(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
def setStdoutToFile(filePath):
    f = open(filePath, 'w')
    sys.stdout = f
    return f
def setStdoutToDefault(f):
    f.close()
    sys.stdout = sys.__stdout__
def doExpBasedExperiments():
    conti = True
    while conti:
        filePath = os.path.join("./experiments.json")
        f = open(filePath)
        exp = json.load(f)
        finishCount = 0
        for expName in exp:
            if exp[expName]==0:
                # exp[expName]=1
                # f = setStdoutToFile("./experiments.json")
                # print(json.dumps(exp, indent=4)) #* make ndarray to list
                # setStdoutToDefault(f)
                subprocess.call('./train.sh')
                break
            finishCount = finishCount + 1
            if finishCount==len(exp):
                exit()
        print("finish trina.sh")
def brutNas():
    # this funciion also handle decode job
    initiManualAssign = {
        # "layer_0_1": [
        #     0,
        #     0,
        #     0,
        #     0,
        #     1
        # ],
        # "layer_1_2": [
        #     0,
        #     1,
        #     0,
        #     0,
        #     0
        # ],
        # "layer_2_3": [
        #     1,
        #     0,
        #     0,
        #     0,
        #     0
        # ],
        "layer_0_4": [
            0,
            0,
            0,
            0,
            0
        ],
        "layer_4_5": [
            0,
            0,
            0,
            0,
            0
        ],
    }
    # brutally train all possible arch of first two layers

    for i in range(5):
        # for fisrt layer
        for j in range(5):
            # for second layer

            manualAssign = copy.deepcopy(initiManualAssign)
            
            manualAssign["layer_0_4"][i] = 1
            manualAssign["layer_4_5"][j] = 1
            f = setStdoutToFile("./curExperiment.json")
            curExpName = "1027_brutL3L4.{}_{}".format(i, j)
            desDir = join("./log", curExpName)
            print(json.dumps({curExpName:1}, indent=4))
            setStdoutToDefault(f)

            makeDir(desDir)
            makeAllDir()
            #info handle decode job
            for kth in range(cfg["numOfKth"]):
                filePath = "./decode/{}th_decode.json".format(kth)
                f = setStdoutToFile(filePath)
                print(json.dumps(manualAssign, indent=4)) #* make ndarray to list
                setStdoutToDefault(f)   
            
            subprocess.call('./train.sh')
            # exit()
            

if __name__=="__main__":
    brutNas()

    

