import subprocess
import json, os, sys

def setStdoutToFile(filePath):
    f = open(filePath, 'w')
    sys.stdout = f
    return f
def setStdoutToDefault(f):
    f.close()
    sys.stdout = sys.__stdout__
if __name__=="__main__":
    # load json

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

