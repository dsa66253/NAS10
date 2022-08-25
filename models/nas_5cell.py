import os
from collections import OrderedDict
from functools import partial
import random
import torch
import torch.nn as nn

from feature.random_seed import set_seed_cpu
from models.alpha.cell import Cell_conv
from data.config import PRIMITIVES, epoch_to_drop, dropNum
import torch.nn.functional as F
import numpy as np

alphas_saveplace = r'./alpha_pdart_nodrop'  # 建立存訓練alpha機率的檔案夾
if not os.path.exists(alphas_saveplace):
    os.makedirs(alphas_saveplace)


class NasModel(nn.Module):
    def __init__(self, num_classes, num_cells, cell=Cell_conv):
        super(NasModel, self).__init__()
        # Initialize alpha
        self.num_cells = num_cells
        self._initialize_alphas()
        self.min_alpha1 = True
        self.min_alpha2 = True
        self.min_alpha3 = True
    
        # 設定nas架構，訓練5個cell
        #* use this way to dynamically determine network
        self.feature = nn.Sequential()
        self.feature.add_module('conv_1', cell(3, 96, 4)) #* inputChannel, outputChannel, stride
        self.feature.add_module('max_pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.feature.add_module('conv_2', cell(96, 256, 1))
        self.feature.add_module('max_pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.feature.add_module('conv_3', cell(256, 384, 1))
        self.feature.add_module('conv_4', cell(384, 384, 1))
        self.feature.add_module('conv_5', cell(384, 256, 1))
        self.feature.add_module('max_pool3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    # 初始化alpha
    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)
        alphas = (1e-3 * torch.ones(self.num_cells, num_ops)) #create tensor with element 1
        alphas = alphas.detach().requires_grad_(True) #*這樣的寫法是用detah() copy alphas，然後要追蹤gradient

        self.alphas_mask = torch.full(alphas.shape, False, dtype=torch.bool)
        self._arch_parameters = [alphas]# * Just want to use zip(), wrap alphas with []
        self._arch_param_names = ['alphas'] #* Just want to use zip(), wrap string with []
        [self.register_parameter(name, nn.Parameter(param))
        for name, param in zip(self._arch_param_names, self._arch_parameters)]

    # @staticmethod
    # 當epoch等於所設定要剔除的epoch位置，將cell中最小的alpha值對應的mask設True
    def check_min_alpah(self, input, epoch):
        if (epoch == epoch_to_drop[0] and self.min_alpha1):
            for n in range(dropNum[0]):#*only one-iteration loop
                print(f'Drop min alphas: {epoch}, {epoch_to_drop[0]}')
                tmp_input_for_min = input[~self.alphas_mask].reshape(self.num_cells, -1)  # ~表示相反

                _, min_indices = tmp_input_for_min.min(1) #* get index of min for each cell
                next_alphas_mask = self.alphas_mask.clone()
                next_mask = torch.full(tmp_input_for_min.shape, False, dtype=torch.bool)
                for i, min_i in zip(range(len(input)), min_indices):  # 把最小的位置變成true
                    next_mask[i, min_i] = True

                next_mask = next_mask.reshape(-1)
                tmp_next_alphas_mask_mask = next_alphas_mask[~self.alphas_mask].clone()#* use previous alphas_mask
                tmp_next_alphas_mask_mask[next_mask] = True #* use previous alphas_mask, and turn specific element to True
                next_alphas_mask[~self.alphas_mask] = tmp_next_alphas_mask_mask
                self.alphas_mask = next_alphas_mask
            self.min_alpha1 = False

        if epoch == epoch_to_drop[1] and self.min_alpha2:
            for n in range(dropNum[1]):
                print(f'Drop min alphas: {epoch}, {epoch_to_drop[1]}')

                tmp_input_for_min = input[~self.alphas_mask].reshape(self.num_cells, -1)  # ~表示相反
                
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
            for n in range(dropNum[2]):
                print(f'Drop min alphas: {epoch}, {epoch_to_drop[2]}')
                tmp_input_for_min = input[~self.alphas_mask].reshape(self.num_cells, -1)  # ~表示相反
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

        with torch.no_grad():
            input[self.alphas_mask] = 0 #* 把是true的alphas設為0
        return input

    # 讓alpha值過softmax變機率值
    #* 這樣的寫法，在一個epoch中，會存很多次不同的alphas值，每一次都覆蓋掉前面一次的
    def normalize(self, input, epoch, number, num_cells):
        # print('----------input data---------------')
        # print(input)
        mask = (input != 0) #* return a tensor that the element is not 0 with element being true false
        new_input = torch.zeros_like(input)
        new_input[mask] += F.softmax(input[mask].reshape(num_cells, -1)).reshape(-1)
        #* new_input[mask] += F.softmax(input[mask].reshape(num_cells, -1), dim=1).reshape(-1) sepcify dim=1 to solve warning

        # print(new_input)

        #save 每個cell中alpha值樣貌
        if epoch % 1 == 0 and epoch > 0:
            alpha_prob = new_input.data.cpu().numpy()
            genotype_filename = os.path.join(alphas_saveplace, 'alpha_prob_' + str(number) + '_' + str(epoch))
            np.save(genotype_filename, alpha_prob)
        return new_input

    def forward(self, x, epoch, number, num_cells):
        count_alpha = 0

        alpha_new = self.check_min_alpah(self.alphas, epoch) #*model's alpha parameters added via registered_parameter()
        norm_alphas = self.normalize(alpha_new, epoch, number, num_cells)
        x = self.feature.conv_1(x, norm_alphas[count_alpha])
        print("x.shape1", x.shape)
        count_alpha += 1
        x = self.feature.max_pool1(x)
        print("x.shapeP1", x.shape)
        x = self.feature.conv_2(x, norm_alphas[count_alpha])
        print("x.shape2", x.shape)
        count_alpha += 1
        x = self.feature.max_pool2(x)
        print("x.shapeP2", x.shape)
        x = self.feature.conv_3(x, norm_alphas[count_alpha])
        print("x.shape3", x.shape)
        count_alpha += 1
        x = self.feature.conv_4(x, norm_alphas[count_alpha])
        print("x.shape4", x.shape)
        count_alpha += 1
        x = self.feature.conv_5(x, norm_alphas[count_alpha])
        print("x.shape5", x.shape)
        count_alpha += 1
        x = self.feature.max_pool3(x)
        # print('check size')
        # print(x.size())
        print("x.shapeP3", x.shape)
        x = x.view(x.size()[0], -1)
        print("x.shape", x.shape)
        x = self.fc(x)
        exit()
        assert count_alpha == len(self.alphas)

        return x
    def nas_parameters(self):
        #* 挑出alpha return出去
        #* parameter name有alpha的都return 出去
        
        if hasattr(self, '_nas_parameters'):
            #* 似乎永遠都不會進來這
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
        #* 挑出weight return出去
        #* parameter name沒alpha的都return 出去
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
