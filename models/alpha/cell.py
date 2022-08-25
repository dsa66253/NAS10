import random
import torch
import torch.nn as nn
import numpy as np

from models.alpha.operation import OPS
from models.alpha.operation_max import OPS_max
from data.config import PRIMITIVES, PRIMITIVES_max, PRIMITIVES_skip


class MixedOp_conv(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super(MixedOp_conv, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](c_in, c_out, stride, False, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(c_out, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        sum = 0
        for w, op in zip(weights, self._ops): 
            if w != 0:
                sum = sum + w * op(x)
        return sum
        # return sum(w * op(x) for w, op in zip(weights, self._ops) if w != 0)


class Cell_conv(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.mix_op = MixedOp_conv(c_in, c_out, stride)

    def forward(self, x, weights):
        y = self.mix_op(x, weights)  # x is input , weights is alpha value
        return y


class MixedOp_skip(nn.Module):

    def __init__(self, c_in, c_out, stride):
        super(MixedOp_skip, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_skip:
            op = OPS[primitive](c_in, c_out, stride, False, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(c_out, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops) if w != 0)


class Cell_skip(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.mix_op = MixedOp_conv(c_in, c_out, stride)

    def forward(self, x, weights):
        y = self.mix_op(x, weights)  # x is input , weights is alpha value
        return y


class MixedOp_max(nn.Module):

    def __init__(self, c_in, c_out, stride):
        super(MixedOp_max, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_max:
            op = OPS_max[primitive](c_in, c_out, stride, False, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(c_out, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops) if w != 0)


class Cell_max(nn.Module):
    def __init__(self, c_in, c_out, stride=2):
        super().__init__()
        self.mix_op = MixedOp_max(c_in, c_out, stride)

    def forward(self, x, weights):
        y = self.mix_op(x, weights)  # x is input , weights is alpha value
        return y
