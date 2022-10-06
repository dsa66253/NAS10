import torch
import torch.nn as nn
import random
import numpy as np
import json, os
def set_seed_cpu(seed):
    # print("set_seed_cpu seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def openCurExp():
    filePath = os.path.join("./curExperiment.json")
    f = open(filePath)
    exp = json.load(f)
    for key in exp:
        return key 

exp2IniFunc = {
    # "r0920_3": lambda weight: torch.nn.init.normal_(weight, 0.025, 0.025/2),
    # "r0920": lambda weight: torch.nn.init.normal_(weight, 0.025, 0.025/2),
    "r0919_3": lambda weight: torch.nn.init.normal_(weight, 0.0125, 0.0125/2),
    "r0919": lambda weight: torch.nn.init.normal_(weight, 0.025, 0.025/2),
    "r0918_3": lambda weight: torch.nn.init.normal_(weight, 0.00625, 0.00625/2),
    "r0918": lambda weight: torch.nn.init.uniform_(weight, 0, 0.025/2),
    "r0924": lambda weight: torch.nn.init.normal_(weight, 0.01, 0.01/2),
    "r0916_4": lambda weight: torch.nn.init.uniform_(weight, -0.05, 0.05),
    "r0916_2": lambda weight: torch.nn.init.uniform_(weight, -0.05/2, 0.05/2),
    "0925": lambda weight: torch.nn.init.uniform_(weight, -0.05, 0.0),
    "0925_3": lambda weight: torch.nn.init.uniform_(weight, 0.0, 0.0025),
    "0926": 0,
    "0927": lambda weight: torch.nn.init.uniform_(weight, 0.025/4, 0.025/4),
    "0927_3": lambda weight: torch.nn.init.uniform_(weight, 0.025/8, 0.025/8),
    "0928_2": lambda weight: torch.nn.init.kaiming_normal_(weight),
    # "0928_3": lambda weight: torch.nn.init.uniform_(weight, -0.05, 0.05),
    # "0928_6": lambda weight: torch.nn.init.uniform_(weight, -0.05/2, 0.05/2),
    "0929": lambda weight: torch.nn.init.uniform_(weight, -0.005/4, 0.005/4),
    "0929_3": lambda weight: torch.nn.init.uniform_(weight, -0.005/8, 0.005/8),
    "r0914_3": lambda weight: torch.nn.init.uniform_(weight, -0.005/2, 0.005/2),
    "r0914_4": lambda weight: torch.nn.init.uniform_(weight, -0.005/4, 0.005/4) 
}
def TsengInitializeWeights(model, seed):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def initialize_weights(model, seed):
    print("set initialize weight with seed ", seed)
    curExp = openCurExp()
    print("cuurent experiment", curExp)
    # TsengInitializeWeights(model, seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            set_seed_cpu(seed)
            # exp2IniFunc[curExp](m.weight)
            # torch.nn.init.kaiming_normal_(m.weight)
            # m.weight = torch.abs(m.weight)
            # torch.nn.init.uniform_(m.weight, 0, 0.025/2)
            # m.weight.data.fill_(0)
            # setTensorPositive(m.weight)
            # torch.nn.init.normal_(m.weight, 0.00625, 0.00625/2)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 1)
        elif isinstance(m, nn.Linear):
            # exp2IniFunc[curExp](m.weight)
            # torch.nn.init.kaiming_normal_(m.weight)
            # setTensorPositive(m.weight.data)
            # torch.nn.init.uniform_(m.weight, 0, 0.025/2)
            # m.weight.data.fill_(0)
            # nn.init.constant_(m.bias, 0)
            # torch.nn.init.normal_(m.weight, 0.00625, 0.00625/2)
            pass
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
def setTensorPositive(tensor):
    tmp = torch.zeros(tensor.shape)
    tmp = torch.nn.init.kaiming_normal_(tmp)
    tmp = torch.abs(tmp)
    with torch.no_grad():
        tensor*=0
        tensor+= tmp
