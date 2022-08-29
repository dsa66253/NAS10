import torch
import torch.nn as nn
from data.config import PRIMITIVES, PRIMITIVES_max, PRIMITIVES_skip
from models.alpha.operation import OPS
from models.alpha.operation_max import OPS_max


class NewMixedOp_conv(nn.Module):
    def __init__(self, c_in, c_out, stride, chosen_op):
        super(NewMixedOp_conv, self).__init__()
        self._ops = nn.ModuleList()
        self.primitive = PRIMITIVES[chosen_op[1]]
        self.op = OPS[self.primitive](c_in, c_out, stride, False, False)

    def forward(self, x):
        return self.op(x)


class NewCell_conv(nn.Module):
    def __init__(self, c_in, c_out, chosen_op, stride=1):
        super(NewCell_conv, self).__init__()
        self.chosen_op = chosen_op
        self.mix_op = NewMixedOp_conv(c_in, c_out, stride, self.chosen_op)

    def forward(self, x):
        y = self.mix_op(x)
        return y


class NewMixedOp_max(nn.Module):
    def __init__(self, c_in, c_out, stride, chosen_op):
        super(NewMixedOp_max, self).__init__()
        self._ops = nn.ModuleList()
        self.primitive = PRIMITIVES_max[chosen_op[1]]
        self.op = OPS_max[self.primitive](c_in, c_out, stride, False, False)

    def forward(self, x):
        return self.op(x)


class NewCell_max(nn.Module):
    def __init__(self, c_in, c_out, chosen_op, stride=1):
        super(NewCell_max, self).__init__()
        self.chosen_op = chosen_op
        self.mix_op = NewMixedOp_max(c_in, c_out, stride, self.chosen_op)

    def forward(self, x):
        y = self.mix_op(x)
        return y


class NewMixedOp_skip(nn.Module):
    def __init__(self, c_in, c_out, stride, chosen_op):
        super(NewMixedOp_skip, self).__init__()
        self._ops = nn.ModuleList()
        self.primitive = PRIMITIVES_skip[chosen_op[1]]
        self.op = OPS[self.primitive](c_in, c_out, stride, False, False)

    def forward(self, x):
        return self.op(x)


class NewCell_skip(nn.Module):
    def __init__(self, c_in, c_out, chosen_op, stride=1):
        super(NewCell_skip, self).__init__()
        self.chosen_op = chosen_op
        self.mix_op = NewMixedOp_skip(c_in, c_out, stride, self.chosen_op)

    def forward(self, x):
        y = self.mix_op(x)
        return y


class NewNasModel(nn.Module):
    def __init__(self, num_classes, cell_arch, num_cells, cell_conv=NewCell_conv, cell_max=NewCell_max,
                 cell_skip=NewCell_skip):
        super(NewNasModel, self).__init__()
        # Initialize alpha
        self.num_cells = num_cells
        self.cell_arch = cell_arch
        # get feature
        num_cell = 0
        self.feature = nn.Sequential()
        self.feature.add_module('op_1', cell_conv(3, 96, cell_arch[num_cell], 4))
        num_cell += 1
        self.feature.add_module('op_2', cell_max(96, 96, cell_arch[num_cell], 2))
        num_cell += 1
        self.feature.add_module('op_3', cell_conv(96, 256, cell_arch[num_cell], 1))
        num_cell += 1
        self.feature.add_module('op_4', cell_max(256, 256, cell_arch[num_cell], 2))
        num_cell += 1
        self.feature.add_module('op_5', cell_conv(256, 384, cell_arch[num_cell], 1))
        num_cell += 1
        self.feature.add_module('op_6', cell_conv(384, 384, cell_arch[num_cell], 1))
        num_cell += 1
        self.feature.add_module('op_7', cell_conv(384, 256, cell_arch[num_cell], 1))
        num_cell += 1
        self.feature.add_module('op_8', cell_max(256, 256, cell_arch[num_cell], 2))

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
