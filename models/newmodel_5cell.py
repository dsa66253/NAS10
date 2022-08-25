import torch
import torch.nn as nn
from data.config import PRIMITIVES
from models.alpha.operation import OPS


class NewMixedOp(nn.Module):
    def __init__(self, c_in, c_out, stride, chosen_op):
        super(NewMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.primitive = PRIMITIVES[chosen_op[1]]
        self.op = OPS[self.primitive](c_in, c_out, stride, False, False)

    def forward(self, x):
        return self.op(x)


class NewCell(nn.Module):
    def __init__(self, c_in, c_out, chosen_op, stride=1):
        super(NewCell, self).__init__()
        self.chosen_op = chosen_op
        self.mix_op = NewMixedOp(c_in, c_out, stride, self.chosen_op)

    def forward(self, x):
        y = self.mix_op(x)
        return y


class NewNasModel(nn.Module):
    def __init__(self, num_classes, cell_arch, num_cells, cell=NewCell):
        super(NewNasModel, self).__init__()
        # Initialize alpha
        self.num_cells = num_cells
        self.cell_arch = cell_arch

        # get feature
        num_cell = 0
        self.feature = nn.Sequential()
        self.feature.add_module('conv_1', cell(3, 96, cell_arch[num_cell], 4))
        num_cell += 1
        self.feature.add_module('max_pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.feature.add_module('conv_2', cell(96, 256, cell_arch[num_cell], 1))
        num_cell += 1
        self.feature.add_module('max_pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.feature.add_module('conv_3', cell(256, 384, cell_arch[num_cell], 1))
        num_cell += 1
        self.feature.add_module('conv_4', cell(384, 384, cell_arch[num_cell], 1))
        num_cell += 1
        self.feature.add_module('conv_5', cell(384, 256, cell_arch[num_cell], 1))

        self.feature.add_module('max_pool3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

        assert num_cell == num_cells - 1, 'Cell number is not consistent!'

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), 256 * 7 * 7)  # 开始全连接层的计算
        x = self.fc(x)

        return x
    
if __name__ == '__main__':
    print(OPS)