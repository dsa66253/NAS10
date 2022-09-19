import torch.nn as nn
import torch
import torch.optim as optim
from . initWeight import initialize_weights
OPS = {
    #'conv_1x1': lambda C_in, C_out, stride, affine, use_ABN: Conv(C_in, C_out, kernelSize, stride, padding, affine=affine),
    'conv_3x3': lambda C_in, C_out, stride, affine, use_ABN: Conv(C_in, C_out, 3, stride, 1, affine=affine),
    'conv_5x5': lambda C_in, C_out, stride, affine, use_ABN: Conv(C_in, C_out, 5, stride, 2, affine=affine),
    'conv_7x7': lambda C_in, C_out, stride, affine, use_ABN: Conv(C_in, C_out, 7, stride, 3, affine=affine),
    'conv_9x9': lambda C_in, C_out, stride, affine, use_ABN: Conv(C_in, C_out, 9, stride, 4, affine=affine),
    'conv_11x11': lambda C_in, C_out, stride, affine, use_ABN: Conv(C_in, C_out, 11, stride, 5, affine=affine),
}


class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )
        self.switch = True #* this conv will be used
        # self.linear = nn.Linear(8, 2)
        self.__initialize_alphas()
        # self.__initialize_weights() #* initialize kernel weights
    def turnSwitch(self, onOrOff):
        if onOrOff==0 or onOrOff==False:
            self.switch = False
        else:
            self.switch = True
    def setAlpha(self, value):
        with torch.no_grad():
            self.alpha *= 0
            self.alpha += value
    def zeroAlpha(self):
        # print("type(self.alpha)", type(self.alpha))
        with torch.no_grad():
            self.alpha *= 0
        # todo drop alpha algo
        self.alpha.requires_grad=False
        self.turnSwitch(False)
    def getAlpha(self):
        return self.alpha
    def getSwitch(self):
        return self.switch
    def __initialize_alphas(self):
        self.alpha = nn.Parameter(torch.FloatTensor([3.14]))
        self.register_parameter( "alpha", self.alpha )
        
    def forward(self, x):
        output = self.op(x)
        # print("input.shape", x.shape)
        # output = self.op(x)*self.alpha
        # print("output.shape", output.shape)
        # output = torch.flatten(output, start_dim=1)
        # print("ouput.shape ", output.shape)
        # output = self.linear(output)
        return output
    def show(self):
        for k, v in self.named_parameters():
            if k in "alpha":
                print(v, v.grad, id(self.alpha))
                
                # print("true+++++++++++")
    def __initialize_weights(self):
        initialize_weights(self)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         if m.weight is not None:
        #             m.weight.data.fill_(1)
        #             m.bias.data.zero_()
                    
    def zero(self):
        with torch.no_grad():
            self.alpha *= 0
            # self.alpha += torch.FloatTensor([0.0])

if __name__=="__main__":

    


    torch.manual_seed(10)
    cretira = nn.CrossEntropyLoss()
    op = OPS["conv_3x3"](3, 2, 1, 1, 1)
    optimizer = optim.SGD(op.parameters(), lr=0.01, momentum=0.05)
    # print(op)
    print("+++++")

    sample = torch.rand((1, 3, 2, 2))
    label = torch.FloatTensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])
    # op.to("cuda")
    sample.to("cuda")
    label.to("cuda")
    
    print( sample.device, label.device )
    for i in range(10):
    # print("=============")
        y = op(sample)
        # print(y.shape, label.shape)
        loss = cretira(y, label)
        loss.backward()
        optimizer.step()
        op.show()

    # op.zero()
    print("=============")
    op.zeroAlpha()
    tmp = []
    # tmp.append(op.parameters())
    for k, v in op.named_parameters():
        # print(k)
        if "alpha" in k:
            pass
            # tmp.append(v)
            tmp.append(v)
        else:
            print("add ", k, type(v))
            # tmp.append(v)
    optimizer = optim.SGD(tmp, lr=0.01, momentum=0.05)

    # op.show()
    print("=============")
    
    for i in range(2):
        # 
        y = op(sample)
        print(y.shape, label.shape)
        loss = cretira(y, label)
        loss.backward()
        optimizer.step()
        op.show()
    # print(id(op.alpha))