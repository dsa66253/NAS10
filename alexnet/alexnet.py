import torch
import torch.nn as nn
import random
import numpy as np
# from ..models.initWeight import initialize_weights

class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
        # print("alexnet", torch.rand((2, 3)))
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 96, 11, stride=4),  # Conv1
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(3, 2),  # Pool1
        #     nn.Conv2d(96, 256, 5, padding=2),  # Conv2
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(3, 2),  # Pool2
        #     nn.Conv2d(256, 384, 3, padding=1),  # Conv3
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 384, 3, padding=1),  # Conv4
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, 3, padding=1),  # Conv5
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(3, 2),  # Pool3
        # )
        self.layerDict = nn.ModuleDict({
            "layer_0":nn.Sequential(nn.Conv2d(3, 96, 11, stride=4),  # Conv1
            nn.ReLU(inplace=True),),
            "max_pool1":nn.MaxPool2d(kernel_size=3, stride=2),
            "layer_1":nn.Sequential(nn.Conv2d(96, 256, 5, padding=2),  # Conv2
            nn.ReLU(inplace=True),),
            "max_pool2": nn.MaxPool2d(kernel_size=3, stride=2),
            "layer_2":nn.Sequential(nn.Conv2d(256, 384, 3, padding=1),  # Conv3
            nn.ReLU(inplace=True),),
            "layer_3":nn.Sequential(nn.Conv2d(384, 384, 3, padding=1),  # Conv4
            nn.ReLU(inplace=True),),
            "layer_4":nn.Sequential(nn.Conv2d(384, 256, 3, padding=1),  # Conv5
            nn.ReLU(inplace=True),),
            'max_pool3': nn.MaxPool2d(kernel_size=3, stride=2)
        })
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # self._initialize_weights()

    def forward(self, input):
        # print("input", x)
        # set_seed_cpu(20)
        # output = self.features(input)
        # print("output", x)
        # x = x.view(x.size(0), 256 * 5 * 5)  # 开始全连接层的计算
        # print("x.shape", x.shape)
        # print("output", x)
        print("input", input.shape)
        output = self.layerDict["layer_0"](input)
        print("layer_0", output.shape)
        output = self.layerDict["max_pool1"](output)
        print("max_pool1", output.shape)
        output = self.layerDict["layer_1"](output)
        print("layer_1", output.shape)
        output = self.layerDict["max_pool2"](output)
        print("max_pool2", output.shape)
        output = self.layerDict["layer_2"](output)
        print("layer_2", output.shape)
        output = self.layerDict["layer_3"](output)
        print("layer_3", output.shape)
        output = self.layerDict["layer_4"](output)
        print("layer_4", output.shape)
        output = self.layerDict["max_pool3"](output)
        output = torch.flatten(output, start_dim=1)
        # print("output", x)
        output = self.classifier(output)
        # print("output", x)
        return output
    def _initialize_weights(self):
        # initialize_weights(self)
        pass
                    
                    
if __name__=="__main__":
    model = Baseline(10)
    input = torch.rand(128, 3, 128, 128)
    model(input)
# class Baseline(nn.Module):
#     def __init__(self, num_classes):
#         super(Baseline, self).__init__()
#         # self.layerDict = nn.ModuleDict({
#         #     "layer_0":nn.Sequential(nn.Conv2d(3, 96, 11, stride=4),  # Conv1
#         #     nn.ReLU(inplace=True),),
#         #     "layer_1":nn.Sequential(nn.Conv2d(96, 256, 5, padding=2),  # Conv2
#         #     nn.ReLU(inplace=True),),
#         #     "layer_2":nn.Sequential(nn.Conv2d(256, 384, 3, padding=1),  # Conv3
#         #     nn.ReLU(inplace=True),),
#         #     "layer_3":nn.Sequential(nn.Conv2d(384, 384, 3, padding=1),  # Conv4
#         #     nn.ReLU(inplace=True),),
#         #     "layer_4":nn.Sequential(nn.Conv2d(384, 256, 3, padding=1),  # Conv5
#         #     nn.ReLU(inplace=True),),
#         #     "max_pool1":nn.MaxPool2d(kernel_size=3, stride=2),
#         #     "max_pool2": nn.MaxPool2d(kernel_size=3, stride=2),
#         #     'max_pool3': nn.MaxPool2d(kernel_size=3, stride=2)
#         # })
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 96, 11, stride=4),  # Conv1
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2),  # Pool1
#             nn.Conv2d(96, 256, 5, padding=2),  # Conv2
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2),  # Pool2
#             nn.Conv2d(256, 384, 3, padding=1),  # Conv3
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 384, 3, padding=1),  # Conv4
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, 3, padding=1),  # Conv5
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2),  # Pool3
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 2 * 2, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#         self._initialize_weights()

#     def forward(self, input):
#         # print("input", x)
#         # set_seed_cpu(20)
#         output = self.features(input)
#         # print("output", x)
#         # x = x.view(x.size(0), 256 * 5 * 5)  # 开始全连接层的计算
#         # print("x.shape", x.shape)
#         # print("output", x)
#         # output = self.layerDict["layer_0"](input)
        
#         # output = self.layerDict["max_pool1"](output)
#         # output = self.layerDict["layer_1"](output)
#         # output = self.layerDict["max_pool2"](output)
#         # output = self.layerDict["layer_2"](output)
#         # output = self.layerDict["layer_3"](output)
#         # output = self.layerDict["layer_4"](output)
#         # output = self.layerDict["max_pool3"](output)
        
#         output = torch.flatten(output, start_dim=1)
#         print("output", output)
#         output = self.classifier(output)
#         # print("output", x)
#         return output
#     def _initialize_weights(self):
#         initialize_weights(self)