# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 13:33:28 2019

@author: Muaz Usmani
"""


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
import matplotlib.cm as cm
import matplotlib

# matplotlib.use("Agg")

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
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import networkx as nx
from collections import OrderedDict
import random

use_zca = False
mean_std = {True: (0.0856, 0.8943), False: (0.2860, 0.3530)}


def parse_args():
    global args
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    #
    parser.add_argument('--data', metavar='DIR', help='path to dataset', default='../data/MNIST/original/train')

    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')

    parser.add_argument('--nmb_cluster', '--k', type=int, default=100,
                        help='number of cluster for k-means (default: 100)')

    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
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
    parser.add_argument('--exp', type=str, default='./exps/fresh/MNIST/100clusters', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty', default=True)

    args = parser.parse_args()


def show_graph_with_labels(adjacency_matrix, labels):
    st = time.time()
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges, k=10)

    # nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    # Add some random weights (as dictonary with edges as key and weight as value).
    # nx.set_edge_attributes(gr, 'my_weight', dict(zip(gr.edges(), [random.random() * 10 for edge in gr.edges()])))

    pos = nx.spring_layout(gr)
    nx.draw(gr, pos=pos, node_size=5, node_color=labels, cmap=plt.cm.Blues)  # , with_labels=True
    print("Took ", time.time() - st)
    plt.show()


def create_adj(lbl_arr):
    row, col, data = [], [], []
    for i in range(len(lbl_arr)):
        row.append(i)
        col.append(i)
        # data.append(lbl_arr[i])
        data.append(1)
        for j in range(i + 1, len(lbl_arr)):
            if lbl_arr[i] == lbl_arr[j]:
                row.append(i)
                col.append(j)
                data.append(1)
                row.append(j)
                col.append(i)
                data.append(1)

    # code to add root node
    unqs = np.unique(lbl_arr)
    frst_idx = []
    for el in unqs:
        frst_idx.append(np.where(lbl_arr == el)[0][0])

    for ex in frst_idx:
        row.append(ex)
        col.append(0)
        data.append(1)

        col.append(ex)
        row.append(0)
        data.append(1)

    adj = csr_matrix((data, (row, col)), shape=(len(lbl_arr), len(lbl_arr))).toarray().astype(np.uint8)
    return adj


def show_proj_features(ftrs, nmb_cluster, pseudo_labels):
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(ftrs)

    area = np.pi * 5
    colors = cm.rainbow(np.linspace(0, 1, nmb_cluster))
    for lbl_id, c in zip(range(nmb_cluster), colors):
        idx = np.where(pseudo_labels == lbl_id)
        # print(len(idx[0]))
        if len(idx[0]) > 1000:
            idx = np.random.choice(idx[0], 1000)
        sel = features_2d[idx]

        x, y = sel[:, 0], sel[:, 1]

        # plot
        plt.scatter(x, y, s=area, color=c, label=str(lbl_id))

    # plot
    plt.legend()
    # ax.legend()
    plt.title('Feature visualization with ' + str(args.nmb_cluster) + " classes")
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.savefig(str(args.nmb_cluster) + " classes_sampled.png")
    plt.show()


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
    # loading model
    state_dict = torch.load(args.resume)
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if not (name == 'top_layer.weight' or name == 'top_layer.bias'):
            model_state[name].copy_(param)
            # print('Copied ', name)
    model = model.to(device)
    tra = [
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=5),
        transforms.ToTensor(),
        transforms.Normalize((mean_std[use_zca][0],), (mean_std[use_zca][1],))
    ]

    dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)

    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # ignoring ReLU layer in classifier

    # get the features for the whole dataset
    features, all_labels = compute_features(dataloader, model, len(dataset),
                                            device)  # ndarray, (60k, 512) [-0.019, 0.016]

    # clustering algorithm to use
    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    # cluster the features
    clustering_loss = deepcluster.cluster(features, verbose=args.verbose)
    # deepcluster.images_lists each row is a cluster id, each column is a image id.

    y = [-1 for i in range(60000)]
    for clstr_id, clstr in enumerate(deepcluster.images_lists):
        for img_id in clstr:
            y[img_id] = clstr_id

    pseudo_labels = np.array(y)  # (60000, )  0, 99

    idx = np.random.randint(pseudo_labels.shape[0], size=1000)

    pseudo_labels = pseudo_labels[idx]
    all_labels = all_labels[idx]

    adj = create_adj(pseudo_labels)

    show_graph_with_labels(adj, all_labels.tolist())

    # show_proj_features(features, args.nmb_cluster, all_labels)

    exit()

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
        for i, (input_tensor, lbls) in enumerate(dataloader):

            # input_tensor torch.Size([256, 1, 28, 28])  tensor(-0.8102) tensor(2.0227)
            input_var = torch.autograd.Variable(input_tensor.to(device))
            aux = model(input_var).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')
                all_labels = lbls.data.numpy()

            else:
                all_labels = np.append(all_labels, lbls.data.numpy(), axis=0)

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
    return features, all_labels


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
