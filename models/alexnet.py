import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, num_classes):
        super(Baseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # Pool1
            nn.Conv2d(96, 256, 5, padding=2),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # Pool2
            nn.Conv2d(256, 384, 3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),  # Pool3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 5 * 5)  # 开始全连接层的计算
        # print("x.shape", x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
