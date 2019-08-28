# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 13:33:28 2019

@author: Muaz Usmani
"""


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from torchvision import datasets, transforms
from torch import nn
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models import VGGMiniCBR
from tensorboardX import SummaryWriter
import os
import argparse
import torch.backends.cudnn as cudnn
from util import AverageMeter, Logger, UnifLabelSampler
import time
import clustering
from sklearn.metrics.cluster import normalized_mutual_info_score

use_zca = False
mean_std = {True: (0.0856, 0.8943), False: (0.2860, 0.3530)}


def parse_args():
    global args
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    #
    parser.add_argument('--data', metavar='DIR', help='path to dataset', default='../data/FashionMNIST/original/train')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10,
                        help='number of cluster for k-means (default: 10)')
    parser.add_argument('--exp', type=str, default='./exps/fresh/FashionMNIST/10clusters', help='path to exp folder')

    parser.add_argument('--epochs', type=int, default=50,
                        help='number of total epochs to run (default: 50)')

    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')




    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--verbose', action='store_true', help='chatty', default=True)

    args = parser.parse_args()


def main():
    global args

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: VGGMiniCBR')

    model = VGGMiniCBR(num_classes=10)

    fd = int(model.top_layer.weight.size()[1])

    model.top_layer = None
    model.to(device)
    cudnn.benchmark = True

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.wd,
    )

    # optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    cluster_log = Logger(os.path.join(exp_path, 'clusters'))

    tra = [
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=5),
        transforms.ToTensor(),
        transforms.Normalize((mean_std[use_zca][0],), (mean_std[use_zca][1],))
    ]

    end = time.time()
    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))

    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()
        # remove head
        model.top_layer = None
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # ignoring ReLU layer in classifier

        # get the features for the whole dataset
        features = compute_features(dataloader, model, len(dataset), device)  # ndarray, (60k, 512) [-0.019, 0.016]

        # cluster the features
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        # assign pseudo-labels
        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                  dataset.imgs)

        # uniformely sample per target
        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        # set last fully connected layer
        mlp = list(model.classifier.children())
        mlp.append(nn.ReLU(inplace=True).to(device))
        model.classifier = nn.Sequential(*mlp)
        model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
        model.top_layer.weight.data.normal_(0, 0.01)
        model.top_layer.bias.data.zero_()
        model.top_layer.to(device)

        # train network with clusters as pseudo-labels
        end = time.time()
        # loss = train(train_dataloader, model, criterion, optimizer, epoch)
        loss = train(model, device, train_dataloader, optimizer, epoch, criterion)

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - end, clustering_loss, loss))
            try:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    clustering.arrange_clustering(cluster_log.data[-1])
                )
                writer.add_scalar('nmi/train', nmi, epoch)
                print('NMI against previous assignment: {0:.3f}'.format(nmi))
            except IndexError:
                pass
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': "VGGMiniCBR",
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(exp_path, 'checkpoint.pth.tar'))

        # save cluster assignments
        cluster_log.log(deepcluster.images_lists)

    torch.save(model.state_dict(), os.path.join(args.exp, "mnist_cnn.pt"))


def train(model, device, train_loader, optimizer, epoch, criterion, scheduler=None, log_interval=50):
    losses = AverageMeter()

    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10 ** args.wd,
    )

    # lr = args.lr,
    #      weight_decay = 10 ** args.wd,
    # optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
    correct = 0
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        # data  tensor(-0.8102) tensor(2.0227) [256, 1, 28, 28]

        data, target = torch.autograd.Variable(data.to(device)), torch.autograd.Variable(target.to(device))

        output = model(data)
        loss = criterion(output, target)

        losses.update(loss.item(), data.size(0))

        optimizer.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        # loss = F.nll_loss(output, target)

        optimizer.step()
        optimizer_tl.step()

        # scheduler.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    writer.add_scalar('loss/train', loss.item(), epoch)
    writer.add_scalar('accuracy/train', 100. * correct / len(train_loader.dataset), epoch)

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss.item(), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

    return losses.avg


def compute_features(dataloader, model, N, device):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    with torch.no_grad():
        for i, (input_tensor, _) in enumerate(dataloader):
            # input_tensor torch.Size([256, 1, 28, 28])  tensor(-0.8102) tensor(2.0227)
            input_var = torch.autograd.Variable(input_tensor.to(device))
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')

            if i < len(dataloader) - 1:
                features[i * args.batch: (i + 1) * args.batch] = aux.astype('float32')
            else:
                # special treatment for final batch
                features[i * args.batch:] = aux.astype('float32')

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if args.verbose and (i % 200) == 0:
            #     print('{0} / {1}\t'
            #           'Time: {batch_time.test:.3f} ({batch_time.avg:.3f})'
            #           .format(i, len(dataloader), batch_time=batch_time))
    return features


if __name__ == '__main__':
    # augmentations:
    # rscale: True
    # rcrop: 712  # random crop of size 713, both dimensions, or can be a tuple.
    # hflip: 0.5  # randomly flip image horizontlly
    # augmentations = {"rscale": True, "rcrop": 712, "hflip": 0.5}
    # data_aug = get_composed_augmentations(augmentations)
    parse_args()
    exp_path = args.exp
    writer = SummaryWriter(log_dir=exp_path)
    main()
