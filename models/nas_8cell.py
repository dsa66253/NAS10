import os
from collections import OrderedDict
from functools import partial
import random
import torch
import torch.nn as nn
from models.alpha.cell import Cell_conv, Cell_max
from data.config import PRIMITIVES_max, epoch_to_drop, dropNum
import torch.nn.functional as F
import numpy as np

alphas_saveplace = r'./alpha_8cell' # 建立存訓練alpha機率的檔案夾
if not os.path.exists(alphas_saveplace):
    os.makedirs(alphas_saveplace)


class NasModel(nn.Module):
    def __init__(self, num_classes, num_cells, cell_conv=Cell_conv, cell_max=Cell_max):
        super(NasModel, self).__init__()
        # Initialize alpha
        self.num_cells = num_cells
        self._initialize_alphas()
        self.min_alpha1 = True
        self.min_alpha2 = True
        self.min_alpha3 = True

        # 設定nas架構，訓練8個cell(有conv跟max cell)
        self.feature = nn.Sequential()
        self.feature.add_module('op_1', cell_conv(3, 96, 4))
        self.feature.add_module('op_2', cell_max(96, 96, 2))
        self.feature.add_module('op_3', cell_conv(96, 256, 1))
        self.feature.add_module('op_4', cell_max(256, 256, 2))
        self.feature.add_module('op_5', cell_conv(256, 384, 1))
        self.feature.add_module('op_6', cell_conv(384, 384, 1))
        self.feature.add_module('op_7', cell_conv(384, 256, 1))
        self.feature.add_module('op_8', cell_max(256, 256, 2))

        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    # 初始化alpha
    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES_max)
        alphas = (1e-3 * torch.ones(self.num_cells, num_ops))

        alpha_max = [1, 3, 7]
        alpha_con_0 = [5, 6]
        alpha_max_0 = [0, 1, 2, 3, 4]

        for i in range(self.num_cells):
            if i not in alpha_max:
                for j in alpha_con_0:
                    # 在conv cell中，使pooling操作的初始設定變超小，使其機率變0，不做訓練
                    alphas[i][j] = -1000
            else:
                for k in alpha_max_0:
                    # 在pooling cell中，使conv操作的初始設定變超小，使其機率變0，不做訓練
                    alphas[i][k] = -1000

        alphas = alphas.detach().requires_grad_(True)

        self.alphas_mask = torch.full(alphas.shape, False, dtype=torch.bool)
        self._arch_parameters = [alphas]
        self._arch_param_names = ['alphas']
        [self.register_parameter(name, nn.Parameter(param))
         for name, param in zip(self._arch_param_names, self._arch_parameters)]

    # @staticmethod
    # 當epoch等於所設定要剔除的epoch位置，將cell中最小的alpha值機率變為0
    def check_min_alpah(self, input, epoch):
        # print(f'check_min_alpah {input}')
        alpha_max = [1, 3, 7]
        alpha_con_0 = [5, 6]
        alpha_max_0 = [0, 1, 2, 3, 4]

        if epoch == epoch_to_drop[0] and self.min_alpha1:
            for i in range(len(self.alphas_mask)):
                for j in alpha_con_0:
                    self.alphas_mask[i][j] = True
                    self.alphas_mask[i][j] = True
            # if self.min_alpha and epoch == epoch_to_drop[0]:
            for n in range(dropNum[0]):
                print(f'Drop min alphas: {epoch}, {epoch_to_drop[0]}')
                tmp_input_for_min = input[~self.alphas_mask].reshape(self.num_cells, -1)  # ~表示相反
                _, min_indices = tmp_input_for_min.min(1)
                # _, min_indices = tmp_input_for_min[tmp_input_for_min > 0].min()
                next_alphas_mask = self.alphas_mask.clone()
                next_mask = torch.full(tmp_input_for_min.shape, False, dtype=torch.bool)
                for i, min_i in zip(range(len(input)), min_indices):  # 把最小的位置變成true
                    next_mask[i, min_i] = True

                next_mask = next_mask.reshape(-1)
                tmp_next_alphas_mask_mask = next_alphas_mask[~self.alphas_mask].clone()
                tmp_next_alphas_mask_mask[next_mask] = True
                next_alphas_mask[~self.alphas_mask] = tmp_next_alphas_mask_mask
                self.alphas_mask = next_alphas_mask
            self.min_alpha1 = False

        if epoch == epoch_to_drop[1] and self.min_alpha2:
            for i in range(len(self.alphas_mask)):
                for j in alpha_con_0:
                    self.alphas_mask[i][j] = True
                    self.alphas_mask[i][j] = True
            for n in range(dropNum[1]):
                print(f'Drop min alphas: {epoch}, {epoch_to_drop[0]}')
                tmp_input_for_min = input[~self.alphas_mask].reshape(self.num_cells, -1)  # ~表示相反
                # import pdb
                # pdb.set_trace()
                _, min_indices = tmp_input_for_min.min(1)

                next_alphas_mask = self.alphas_mask.clone()
                next_mask = torch.full(tmp_input_for_min.shape, False, dtype=torch.bool)
                for i, min_i in zip(range(len(input)), min_indices):  # 把最小的位置變成true
                    next_mask[i, min_i] = True

                next_mask = next_mask.reshape(-1)
                tmp_next_alphas_mask_mask = next_alphas_mask[~self.alphas_mask].clone()
                tmp_next_alphas_mask_mask[next_mask] = True
                next_alphas_mask[~self.alphas_mask] = tmp_next_alphas_mask_mask
                self.alphas_mask = next_alphas_mask
            self.min_alpha2 = False

        if epoch == epoch_to_drop[2] and self.min_alpha3:
            for i in range(len(self.alphas_mask)):
                for j in alpha_con_0:
                    self.alphas_mask[i][j] = True
                    self.alphas_mask[i][j] = True
            for n in range(dropNum[2]):
                print(f'Drop min alphas: {epoch}, {epoch_to_drop[0]}')
                tmp_input_for_min = input[~self.alphas_mask].reshape(self.num_cells, -1)  # ~表示相反
                # import pdb
                # pdb.set_trace()
                _, min_indices = tmp_input_for_min.min(1)

                next_alphas_mask = self.alphas_mask.clone()
                next_mask = torch.full(tmp_input_for_min.shape, False, dtype=torch.bool)
                for i, min_i in zip(range(len(input)), min_indices):  # 把最小的位置變成true
                    next_mask[i, min_i] = True

                next_mask = next_mask.reshape(-1)
                tmp_next_alphas_mask_mask = next_alphas_mask[~self.alphas_mask].clone()
                tmp_next_alphas_mask_mask[next_mask] = True
                next_alphas_mask[~self.alphas_mask] = tmp_next_alphas_mask_mask
                self.alphas_mask = next_alphas_mask
            self.min_alpha3 = False

        for i in range(len(self.alphas_mask)):
            for j in alpha_con_0:
                self.alphas_mask[i][j] = False
                self.alphas_mask[i][j] = False
        with torch.no_grad():
            input[self.alphas_mask] = 0  # 將true的位置的alpha值變成0

        return input

    # 讓alpha值過softmax變機率值
    def normalize(self, input, epoch, number, num_cells):
        # input = torch.rand(5, 5)
        dropItem = 0
        checkDrop = 0
        print('----------input data---------------')
        print(input)

        alpha_max = [1, 3, 7]
        alpha_con_0 = [5, 6]
        alpha_max_0 = [0, 1, 2, 3, 4]

        for item in epoch_to_drop:
            if epoch > item or epoch == item:
                dropItem += 1
            else:
                continue

        with torch.no_grad():
            for i in range(self.num_cells):
                if i not in alpha_max:
                    for j in alpha_con_0:
                        input[i][j] = -1000
                else:
                    for k in alpha_max_0:
                        input[i][k] = -1000

        mask = (input != 0)
        for item in mask[0]:
            if not item:
                checkDrop += 1

        if checkDrop == dropItem:
            with torch.no_grad():
                for i in range(self.num_cells):
                    if i in alpha_max:
                        for k in range(dropItem):
                            input[i][k] = 0
        mask = (input != 0)
        new_input = torch.zeros_like(input)

        new_input[mask] += F.softmax(input[mask].reshape(num_cells, -1)).reshape(-1)

        # save 每個cell中alpha值樣貌
        if epoch % 1 == 0 and epoch > 0:
            alpha_prob = new_input.data.cpu().numpy()
            genotype_filename = os.path.join(alphas_saveplace, 'alpha_prob_' + str(number) + '_' + str(epoch))
            np.save(genotype_filename, alpha_prob)

        return new_input

    def forward(self, x, epoch, number, num_cells):
        count_alpha = 0
        temp_norm_alphas = self.normalize(self.alphas, epoch, number, num_cells)
        fake_norm_alphas = self.check_min_alpah(temp_norm_alphas, epoch)
        norm_alphas = self.normalize(fake_norm_alphas, epoch, number, num_cells)

        x = self.feature.op_1(x, norm_alphas[count_alpha])
        count_alpha += 1
        x = self.feature.op_2(x, norm_alphas[count_alpha])
        count_alpha += 1
        x = self.feature.op_3(x, norm_alphas[count_alpha])
        count_alpha += 1
        x = self.feature.op_4(x, norm_alphas[count_alpha])
        count_alpha += 1
        x = self.feature.op_5(x, norm_alphas[count_alpha])
        count_alpha += 1
        x = self.feature.op_6(x, norm_alphas[count_alpha])
        count_alpha += 1
        x = self.feature.op_7(x, norm_alphas[count_alpha])
        count_alpha += 1
        x = self.feature.op_8(x, norm_alphas[count_alpha])
        count_alpha += 1
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        assert count_alpha == len(self.alphas)

        return x

    def nas_parameters(self):
        if hasattr(self, '_nas_parameters'):
            return self._nas_parameters

        def check_key(k):
            for ele in self._arch_param_names:
                if ele in k:
                    return True
            return False

        self._nas_parameters = {
            v for k, v in self.named_parameters() if check_key(k)
        }
        return self._nas_parameters

    def model_parameters(self):
        if hasattr(self, '_model_parameters'):
            return self._model_parameters

        def check_key(k):
            for ele in self._arch_param_names:
                if ele in k:
                    return False
            return True

        self._model_parameters = {
            v for k, v in self.named_parameters() if check_key(k)
        }
        return self._model_parameters
