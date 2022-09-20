import torch
import torch.nn as nn
import random
import numpy as np
def set_seed_cpu(seed):
    # print("set_seed_cpu seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def initialize_weights(model, seed):
    for m in model.modules():

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            set_seed_cpu(seed)
            # torch.nn.init.kaiming_normal_(m.weight)
            # m.weight = torch.abs(m.weight)
            # torch.nn.init.uniform_(m.weight, 0, 0.025/2)
            # m.weight.data.fill_(0)
            # setTensorPositive(m.weight)
            torch.nn.init.normal_(m.weight, 0.0125, 0.0125/2)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 1)
        elif isinstance(m, nn.Linear):
            # torch.nn.init.kaiming_normal_(m.weight)
            # setTensorPositive(m.weight.data)
            # torch.nn.init.uniform_(m.weight, 0, 0.025/2)
            # m.weight.data.fill_(0)
            # nn.init.constant_(m.bias, 0)
            torch.nn.init.normal_(m.weight, 0.0125, 0.0125/2)
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
