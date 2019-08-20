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
import models
from sklearn.metrics import classification_report
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import os
from utils import save_zca
import argparse

max_entropy = torch.log(torch.tensor(10.0))
percentile = 0.15
threshold = percentile * max_entropy
threshold = threshold.data.numpy()


# mean : tensor(0.2860)  std:  tensor(0.3530) statistics of FashionMNIST
# with zca mean : tensor(0.0856)  std:  tensor(0.8943) statistics of FashionMNIST

def parse_args():
    global args
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--model', type=str, choices=['VGGMiniCBR', 'VGGMini'],
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

    parser.add_argument('--bayes', type=int, default=0,
                        help='To use MCDropout with args.bayes samples. default(0 or not using)')
    args = parser.parse_args()


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


def apply_dropout(m):
    if type(m) == nn.modules.dropout.Dropout2d:
        m.train()


def get_masked_pred(mean_pred):
    '''
    :param tnsr: shape (B, C, W, H)
    :return: entropy vector of shape (BxWxH,)
    '''
    # mean_pred = torch.mean(tnsr, dim=0)  # (B, C, W, H)

    mean_pred = mean_pred.permute(1, 0, 2, 3)  # (C, B, W, H) -4.1693, 4.6553
    # print('mean_pred: ', mean_pred.shape)
    # mean_pred = mean_pred #.contiguous().view(C, -1)  # (C, BxWxH)

    mean_probs = torch.nn.Softmax(dim=0)(mean_pred)  # 0.0005, 0.7798, [19, 10, 28, 28]
    maxed_pred = mean_probs.data.max(0)[1].numpy()  # (10, 28, 28), 0, 18 --> (BS, 1)

    # print('Probs ', mean_probs.size(), mean_probs.min(), mean_probs.max())
    log_probs = torch.log(mean_probs)  # [19, 10, 28, 28], -7.6821, -0.2487-> [10, BS, 1]
    # print('logs ', log_probs.size(), log_probs.min(), log_probs.max())
    mult = mean_probs * log_probs  # [19, 10, 28, 28], -0.3679, -0.0035 -> [10, BS, 1]

    entropy = -torch.sum(mult, dim=0).numpy()  # (10, 28, 28), 0.9850448, 2.9022176 -> [BS, 1]

    preds = {}
    # for th in thresholds_arr:
    mask = np.where(entropy <= threshold, 1, 0)  # (10, 28, 28), 0, 0-> [BS, 1]
    # non_zeros = np.count_nonzero(mask)

    masked_pred = mask * maxed_pred
    # preds[str(th)] = masked_pred
    #
    # preds[str(th) + "_accepted"] = non_zeros

    return masked_pred  # , non_zeros


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

        masked_pred = get_masked_pred(model_out)
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
        # score = 0
        if score > best_pred:
            best_pred = score
            torch.save(model.state_dict(), os.path.join(args.exp, "mnist_cnn.pt"))
            print("Model saved at ", os.path.join(args.exp, "mnist_cnn.pt"))
    cls_report(model, device, test_loader)


if __name__ == '__main__':
    # augmentations:
    # rscale: True
    # rcrop: 712  # random crop of size 713, both dimensions, or can be a tuple.
    # hflip: 0.5  # randomly flip image horizontlly
    # augmentations = {"rscale": True, "rcrop": 712, "hflip": 0.5}
    # data_aug = get_composed_augmentations(augmentations)
    parse_args()

    model_name = args.model.lower()
    augment = "noaugment"
    if args.augment:
        augment = "augment"
    bayes = "nobayes"
    if args.bayes:
        bayes = "bayes" + str(args.bayes)
    strr = model_name + '_' + augment + '_' + bayes

    args.exp = os.path.join(args.exp, strr)
    writer = SummaryWriter(log_dir=args.exp)
    main()
