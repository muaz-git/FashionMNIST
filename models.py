# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:46:14 2019

@author: Muaz Usmani
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


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
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    m = VGGMini(num_classes=10)
    m.cuda(0)

    x = torch.randn(5, 3, 28, 28, device='cuda:0', dtype=torch.float32)
    m(x)
