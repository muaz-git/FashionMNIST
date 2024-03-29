# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 12:46:14 2019

@author: Muaz Usmani
"""

import numpy as np
import torch

from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
import models
from models import ResidualBlock
from sklearn.metrics import classification_report
from tensorboardX import SummaryWriter
import os
from utils import save_zca
import argparse
from utils import apply_dropout, get_masked_pred
from matplotlib import pyplot as plt

def parse_args():
    global args
    parser = argparse.ArgumentParser(description='PyTorch Implementation on FashionMNIST.')

    parser.add_argument('--model', type=str, choices=['VGGMiniCBR', 'VGGMini', 'FashionMNISTResNet', 'VGGMiniRes'],
                        default='VGGMiniCBR', help='Choice of model (default: VGGMiniCBR)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--valbatch', default=1000, type=int,
                        help='mini-batch size for validation (default: 1000)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of total epochs to run (default: 25)')
    parser.add_argument('--exp', type=str, default='./exps/',
                        help='path to exp folder (default: ./exps/)')
    parser.add_argument('--augment', action='store_true', help='To augment data', default=False)
    parser.add_argument('--translate', action='store_true', help='To translate only', default=False)

    parser.add_argument('--bayes', type=int, default=0,
                        help='To use MCDropout with args.bayes samples. default(0 for not using)')
    parser.add_argument('--percentile', type=float, default=0.25,
                        help='When using MCDropout, percentile threshold is used to ignore predicitons where entropy is higher than a particular threshold. (default 0.25)')
    args = parser.parse_args()


parse_args()
max_entropy = torch.log(torch.tensor(10.0))
percentile = args.percentile
threshold = percentile * max_entropy
threshold = threshold.data.numpy()

# mean : tensor(0.2860)  std:  tensor(0.3530) statistics of FashionMNIST
# with zca mean : tensor(0.0856)  std:  tensor(0.8943) statistics of FashionMNIST


labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# fcn = lambda step: 1. / (1. + INIT_LR / NUM_EPOCHS * step)
use_zca = False
if not os.path.exists("./statistics"):
    os.makedirs("./statistics")
if not os.path.exists('./statistics/fashionmnist_zca_3.pt') and use_zca:
    save_zca()
if use_zca:
    W = torch.load('./statistics/fashionmnist_zca_3.pt')

mean_std = {True: (0.0856, 0.8943), False: (0.2860, 0.3530)}

criterion = nn.CrossEntropyLoss()

best_pred = -100.0


# print((mean_std[use_zca][0],), (mean_std[use_zca][1],))


def train(model, device, train_loader, optimizer, epoch, log_interval, scheduler=None):
    global args
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_zca:
            data = torch.matmul(data.reshape((args.batch, 1 * 28 * 28)), W)
            data = data.reshape((args.batch, 1, 28, 28))

        # data tensor(-0.8102) tensor(2.0227) torch.Size([256, 1, 28, 28])

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()

        if scheduler:
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

            if use_zca:
                data = torch.matmul(data.reshape((args.valbatch, 1 * 28 * 28)), W)
                data = data.reshape((args.valbatch, 1, 28, 28))

            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # _, pred = torch.max(output.data, 1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    writer.add_scalar('loss/test', test_loss, epoch - 1)
    writer.add_scalar('accuracy/test', 100. * correct / len(test_loader.dataset), epoch - 1)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def test_mcdropout(model, device, test_loader, epoch):
    # this function follows https://arxiv.org/pdf/1506.02142.pdf to generate distribution of classes instead of point-estimating
    model.eval()
    print('Applying MC Dropout')
    model.apply(apply_dropout)

    n_samples = args.bayes
    test_loss = 0
    correct = 0
    print('n_samples : {}'.format(n_samples))
    for data, target in test_loader:
        if use_zca:
            data = torch.matmul(data.reshape((args.valbatch, 1 * 28 * 28)), W)
            data = data.reshape((args.valbatch, 1, 28, 28))

        data, target = data.to(device), target.to(device)

        model_out = None
        for q in range(n_samples):
            with torch.no_grad():
                if model_out is None:
                    model_out = model(data).detach().data
                else:
                    model_out = model_out + model(data).detach().data

        model_out = model_out / n_samples

        output = F.log_softmax(model_out, dim=1)
        output = output.to(device)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        # masking predictions where entropy is high
        model_out = model_out.cpu()  # [1000, 10] --> [BS, C] -> shape (B, C, W, H)
        model_out = model_out.unsqueeze(2).unsqueeze(3)

        masked_pred, _ = get_masked_pred(model_out, threshold)
        masked_pred = masked_pred.squeeze()  # (1000, )
        masked_pred = torch.from_numpy(masked_pred).to(device)

        # output = F.log_softmax(model_out, dim=1)
        # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        # # _, pred = torch.max(output.data, 1)
        correct += masked_pred.eq(target.view_as(masked_pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    writer.add_scalar('loss/test', test_loss, epoch - 1)
    writer.add_scalar('accuracy/test', 100. * correct / len(test_loader.dataset), epoch - 1)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def cls_report_mcdropout(model, device, test_loader):
    model.eval()
    all_targ = []
    all_pred = []

    n_samples = args.bayes
    print('n_samples : {}'.format(n_samples))

    for data, target in test_loader:
        if use_zca:
            data = torch.matmul(data.reshape((args.valbatch, 1 * 28 * 28)), W)
            data = data.reshape((args.valbatch, 1, 28, 28))

        data = data.to(device)

        model_out = None
        for q in range(n_samples):
            with torch.no_grad():
                if model_out is None:
                    model_out = model(data).detach().data
                else:
                    model_out = model_out + model(data).detach().data

        model_out = model_out / n_samples

        # masking predictions where entropy is high
        model_out = model_out.cpu()  # [1000, 10] --> [BS, C] -> shape (B, C, W, H)
        model_out = model_out.unsqueeze(2).unsqueeze(3)

        masked_pred, _ = get_masked_pred(model_out, threshold)
        masked_pred = masked_pred.squeeze()  # (1000, )
        masked_pred = torch.from_numpy(masked_pred).to(device)

        pred = list(np.squeeze(masked_pred.cpu().detach().numpy()))
        target = list(target.numpy())

        all_targ += target
        all_pred += pred

    print("evaluating network...")
    print(classification_report(all_targ, all_pred,
                                target_names=labelNames))


def cls_report(model, device, test_loader):
    model.eval()
    all_targ = []
    all_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            if use_zca:
                data = torch.matmul(data.reshape((args.valbatch, 1 * 28 * 28)), W)
                data = data.reshape((args.valbatch, 1, 28, 28))

            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # _, pred = torch.max(output.data, 1)
            pred = list(np.squeeze(pred.cpu().detach().numpy()))
            target = list(target.numpy())

            all_targ += target
            all_pred += pred

    print("evaluating network...")
    print(classification_report(all_targ, all_pred,
                                target_names=labelNames))


def main():
    global best_pred, criterion
    use_cuda = torch.cuda.is_available()

    # fix random seeds
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    basic_transform = [transforms.ToTensor(),
                       transforms.Normalize((mean_std[use_zca][0],), (mean_std[use_zca][1],))]
    if args.augment:
        print('Augmenting training data')
        train_transform = [transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05),
                                                   shear=5)] + basic_transform
    #
    elif args.translate:
        train_transform = [transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.999, 1.001),
                                                   shear=0)] + basic_transform
    else:
        print('Not Augmenting training data')
        train_transform = basic_transform

    # Download and load the training data
    # Download and load the test data
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data/', train=True, download=True, transform=transforms.Compose(train_transform)),
        batch_size=args.batch, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data/', train=False, download=True, transform=transforms.Compose(basic_transform)),
        batch_size=args.valbatch, shuffle=True, **kwargs)

    # model = VGGMiniCBR(num_classes=10)
    if args.model == 'VGGMiniRes':
        net_args = {
            "block": ResidualBlock,
            "layers": [2, 2, 2, 2]
        }
        model = models.__dict__[args.model](**net_args, num_classes=10)
    else:
        model = models.__dict__[args.model](num_classes=10)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9)
    # scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    for epoch in range(1, args.epochs + 1):

        train(model, device, train_loader, optimizer, epoch, log_interval=50)
        if args.bayes > 0:
            score = test_mcdropout(model, device, test_loader, epoch)
        else:
            score = test(model, device, test_loader, epoch)

        if score > best_pred:
            best_pred = score
            torch.save(model.state_dict(), os.path.join(args.exp, "mnist_cnn.pt"))
            print("Model saved at ", os.path.join(args.exp, "mnist_cnn.pt"))
    if args.bayes:
        cls_report_mcdropout(model, device, test_loader)
    else:
        cls_report(model, device, test_loader)


if __name__ == '__main__':
    # augmentations:
    # rscale: True
    # rcrop: 712  # random crop of size 713, both dimensions, or can be a tuple.
    # hflip: 0.5  # randomly flip image horizontlly
    # augmentations = {"rscale": True, "rcrop": 712, "hflip": 0.5}
    # data_aug = get_composed_augmentations(augmentations)

    model_name = args.model.lower()
    augment = "noaugment"
    if args.augment:
        augment = "augment"
    if args.translate:
        augment = "translate"
    bayes = "nobayes"
    if args.bayes:
        bayes = "bayes" + str(args.bayes) + '_per' + str(percentile)
    strr = model_name + '_' + augment + '_' + bayes

    args.exp = os.path.join(args.exp, strr)
    writer = SummaryWriter(log_dir=args.exp)
    main()
