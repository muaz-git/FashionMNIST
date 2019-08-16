import torchvision.models
import torch.nn as nn
import torch

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class VGGMini(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGMini, self).__init__()

        self.features = nn.Sequential(
            # first CONV => RELU => CONV => RELU => POOL layer set
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),

            # second CONV => RELU => CONV => RELU => POOL layer set
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(0.25),

        )
        # first (and only) set of FC => RELU layers
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=x, out_features=512),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(0.5),
        #
        #     nn.Linear(in_features=512, out_features=num_classes),
        # )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        # return x


if __name__ == '__main__':
    m = VGGMini(num_classes=10)
    m.cuda(0)

    x = torch.randn(1, 3, 100, 100, device='cuda:0', dtype=torch.float32)
    m(x)