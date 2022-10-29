import numpy as np
import csv
import os
class AccCollector():
    def __init__(self):
        # self.baseDir = "./log/1024_brutL3L4/1024_brutL3L4.0_0/accLoss" 
        # base = os.walk(self.baseDir)
        a = []
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    baseDir = "./log/1024_brutL3L4/1024_brutL3L4.{}_{}/accLoss/retrain_test_acc_{}.npy".format(str(i), str(j), str(k)) 
                    # base = os.walk(baseDir)
                    a.append(np.load(baseDir)[-1])
        print(len(a))
        # for i in base:
        #     print(i)
if __name__=="__main__":
    accC = AccCollector()
    