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
import math
from torchvision.models.resnet import ResNet, BasicBlock

__all__ = ['VGGMini', 'VGGMiniCBR', 'VGGMiniRes', 'FashionMNISTResNet']


class VGGMini(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGMini, self).__init__()
        print("Constructing VGGMini")
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
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 26 * 26, out_features=512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        # first (and only) set of FC => RELU layers

        self.top_layer = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # exit()
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x
        # return F.log_softmax(x, dim=1)


class VGGMiniCBR(nn.Module):  # overall graph is above than CRB method.
    def __init__(self, num_classes=10):
        super(VGGMiniCBR, self).__init__()
        print("Constructing VGGMiniCBR")
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
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
        )
        self.top_layer = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)

        # if self.training and self.aux:

        # print(x.shape)
        # exit()
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x
        # return F.log_softmax(x, dim=1)


# conv3x3 and ResidualBlock code from https://www.kaggle.com/readilen/resnet-for-mnist-with-pytorch
# 3*3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class VGGMiniRes(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(VGGMiniRes, self).__init__()

        self.in_channels = 16
        self.first_mod = nn.Sequential(
            conv3x3(1, 16),
            nn.BatchNorm2d(16)
        )

        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)

        self.avg_pool = nn.AvgPool2d(8)
        self.top_layer = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.first_mod(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)

        out = out.view(out.size(0), -1)
        out = self.top_layer(out)
        return out


class FashionMNISTResNet(ResNet):
    def __init__(self, num_classes=10):
        super(FashionMNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)

    def forward(self, x):
        return torch.softmax(
            super(FashionMNISTResNet, self).forward(x), dim=-1)


def get_n_params(model):
    def millify(n):
        millnames = ['', ' Thousand', ' Million', ' Billion', ' Trillion']
        n = float(n)
        millidx = max(0, min(len(millnames) - 1,
                             int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

        return '{:.0f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])

    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return millify(pp)


if __name__ == '__main__':
    # Create ResNet
    net_args = {
        "block": ResidualBlock,
        "layers": [2, 2, 2, 2]
    }
    m = FashionMNISTResNet() # 0.005, 195k params
    print(m)
    print(get_n_params(m))
    exit()
    # m = VGGMini(num_classes=10)  # 0.00263 for 32 imgs 1 channel
    # m = VGGMiniCBR(num_classes=10)  # 0.00255 for 32 imgs 1 channel
    # x = get_n_params(m)
    # print(x)
    # exit()
    # m = models.resnet18(pretrained=True)  # 0.0299 for 32 imgs of 3 channles
    # m = models.resnet34(pretrained=True)  # 0.0442 for 32 imgs of 3 channles
    # model = WideResNet(depth=40, num_classes=10) # Average: 0.0071  for 32 imgs of 3 channles  0.007
    # m = models.mobilenet_v2(pretrained=False)  # 0.05622  for 32 imgs of 3 channles
    # m = models.
    # print(m)
    # exit()
    m.cuda(0)
    ts = []
    x = torch.randn(32, 1, 28, 28, device='cuda:0', dtype=torch.float32)
    for t in range(100):
        et = time.time()
        m(x)

        ts.append(time.time() - et)

    print('Average :', np.average(ts))

    exit()
    # print(m)
