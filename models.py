# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:46:14 2019

@author: Muaz Usmani
"""

import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import time
import numpy as np


class VGGMini(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGMini, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
        #                 padding_mode='zeros')
        self.features = nn.Sequential(
            # first CONV => RELU => CONV => RELU => POOL layer set
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),

            # second CONV => RELU => CONV => RELU => POOL layer set
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),

        )
        # first (and only) set of FC => RELU layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 26 * 26, out_features=512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # exit()
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class VGGMiniCBR(nn.Module): # overall graph is above than CRB method.
    def __init__(self, num_classes=10):
        super(VGGMiniCBR, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
        #                 padding_mode='zeros')
        self.features = nn.Sequential(
            # first CONV => RELU => CONV => RELU => POOL layer set
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),

            # second CONV => RELU => CONV => RELU => POOL layer set
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),

        )
        # first (and only) set of FC => RELU layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 26 * 26, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # exit()
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    # m = VGGMini(num_classes=10) # 0.01675469160079956 for 32 imgs 1 channel
    # m = models.resnet18(pretrained=True)  # 0.0299 for 32 imgs of 3 channles
    # m = models.resnet34(pretrained=True)  # 0.0442 for 32 imgs of 3 channles
    m = models.mobilenet_v2(pretrained=False)  # 0.05622  for 32 imgs of 3 channles
    # print(m)
    # exit()
    m.cuda(0)
    ts = []
    x = torch.randn(32, 3, 28, 28, device='cuda:0', dtype=torch.float32)
    for t in range(100):
        et = time.time()
        m(x)
        ts.append(time.time() - et)

    print('Average :', np.average(ts))

    # print(m)
