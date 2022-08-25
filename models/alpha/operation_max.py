import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
import numpy as np

# TODO: NOW I DONT KNOW HOW TO USE ABN ON WINDOWS SYSTEM

if platform.system() == 'Windows':

    class ABN(nn.Module):
        def __init__(self, C_out, affine=False):
            super(ABN, self).__init__()
            self.op = nn.Sequential(
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False)
            )

        def forward(self, x):
            return self.op(x)
else:
    # Not implemented
    # from models.alpha.bn import InPlaceABNSync as ABN
    pass

OPS_max = {
    'max_pool_3x3': lambda C_in, C_out, stride, affine, use_ABN: nn.MaxPool2d(3, stride=stride, padding=1),
    'avg_pool_3x3': lambda C_in, C_out, stride, affine, use_ABN: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'skip_connect': lambda C_in, C_out, stride, affine, use_ABN: Identity(C_in, C_out) if stride == 1 else
    (FactorizedReduce(C_in, C_out, affine=affine) if stride == 2
     else DoubleFactorizedReduce(C_in, C_out, affine=affine))
}


class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )
        self._initialize_weights()

    def forward(self, x):
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


class Maxpooling(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Maxpooling, self).__init__()
        self.op = nn.Sequential(
           #nn.ReLU(inplace=False),
            #nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(C_out, affine=False),
            nn.MaxPool2d(3, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self, C_in, C_out):
        super(Identity, self).__init__()

        if C_in != C_out:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=False)
            )
        else:
            self.op = nn.Sequential()

        self._initialize_weights()

    def forward(self, x):
        # return x
        return self.op(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self._initialize_weights()

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


class FactorizedReduce(nn.Module):
    # TODO: why conv1 and conv2 in two parts ?
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


class DoubleFactorizedReduce(nn.Module):
    # TODO: why conv1 and conv2 in two parts ?
    def __init__(self, C_in, C_out, affine=True):
        super(DoubleFactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=4, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
