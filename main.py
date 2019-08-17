# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:46:14 2019

@author: Muaz Usmani
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from models import VGGMiniCBR
from sklearn.metrics import classification_report
from tensorboardX import SummaryWriter

from torch.optim.lr_scheduler import LambdaLR

BS = 32  # 256
INIT_LR = 1e-2
NUM_EPOCHS = 25

# mean : tensor(0.2860)  std:  tensor(0.3530) statistics of FashionMNIST

labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]

iterr = 0
fcn = lambda step: 1. / (1. + INIT_LR / NUM_EPOCHS * step)


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = np.squeeze(image.numpy())

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def view_classify(img, ps, version="Fashion"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                             'Trouser',
                             'Pullover',
                             'Dress',
                             'Coat',
                             'Sandal',
                             'Shirt',
                             'Sneaker',
                             'Bag',
                             'Ankle Boot'], size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)


def train(model, device, train_loader, optimizer, scheduler, epoch, log_interval):
    global iterr
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        iterr += 1
        optimizer.step()

        scheduler.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    writer.add_scalar('loss/train', loss.item(), epoch - 1)
    writer.add_scalar('accuracy/train', 100. * correct / len(train_loader.dataset), epoch - 1)

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss.item(), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    writer.add_scalar('loss/val', test_loss, epoch - 1)
    writer.add_scalar('accuracy/val', 100. * correct / len(test_loader.dataset), epoch - 1)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def cls_report(model, device, test_loader):
    model.eval()
    all_targ = []
    all_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = list(np.squeeze(pred.cpu().detach().numpy()))
            target = list(target.numpy())

            all_targ += target
            all_pred += pred

    print("evaluating network...")
    print(classification_report(all_targ, all_pred,
                                target_names=labelNames))


def main():
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Download and load the training data
    # Download and load the test data
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data/', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05),
                                                          shear=5),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.2860,), (0.3530,))
                              ])),
        batch_size=BS, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])),
        batch_size=1000, shuffle=True, **kwargs)

    image, _ = next(iter(train_loader))

    model = VGGMiniCBR(num_classes=10)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, device, train_loader, optimizer, scheduler, epoch, log_interval=100)
        test(model, device, test_loader, epoch)

    cls_report(model, device, test_loader)
    # torch.save(model.state_dict(), "fashionmnist_cnn_wo_lastconv.pt")


if __name__ == '__main__':
    # augmentations:
    # rscale: True
    # rcrop: 712  # random crop of size 713, both dimensions, or can be a tuple.
    # hflip: 0.5  # randomly flip image horizontlly
    # augmentations = {"rscale": True, "rcrop": 712, "hflip": 0.5}
    # data_aug = get_composed_augmentations(augmentations)
    exp_path = "./exps/25epochs/Aug/moderateFlip"
    writer = SummaryWriter(log_dir=exp_path)
    main()
