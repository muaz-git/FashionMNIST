import argparse
from tensorboardX import SummaryWriter
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import classification_report

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from models import VGGMiniCBR

use_zca = False
mean_std = {True: (0.0856, 0.8943), False: (0.2860, 0.3530)}

criterion = nn.CrossEntropyLoss()

labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]


def parse_args():
    global args
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    #
    # parser.add_argument('data', metavar='DIR', help='path to dataset', default='../data/FashionMNIST/original/train')

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
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--valbatch', default=1000, type=int,
                        help='mini-batch size (default: 1000)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--exp', type=str, default='./exps/eval', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty', default=True)

    args = parser.parse_args()


def main():
    global args

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = VGGMiniCBR(num_classes=10)
    model.to(device)

    criterion.to(device)

    cudnn.benchmark = True

    # freeze the features layers
    for param in model.features.parameters():
        param.requires_grad = False

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data/', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05),
                                                          shear=5),
                                  # transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((mean_std[use_zca][0],), (mean_std[use_zca][1],))
                              ])),
        batch_size=args.batch, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean_std[use_zca][0],), (mean_std[use_zca][1],))
        ])),
        batch_size=args.valbatch, shuffle=True, **kwargs)

    optimizer = optim.Adam(model.parameters())

    for epoch in range(0, args.epochs):
        train(model, device, train_loader, optimizer, None, epoch, log_interval=50)
        test(model, device, test_loader, epoch)

    cls_report(model, device, test_loader)


def train(model, device, train_loader, optimizer, scheduler, epoch, log_interval):
    global args
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        # data tensor(-0.8102) tensor(2.0227) torch.Size([256, 1, 28, 28])

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()

        # scheduler.step()

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
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # _, pred = torch.max(output.data, 1)
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
            # _, pred = torch.max(output.data, 1)
            pred = list(np.squeeze(pred.cpu().detach().numpy()))
            target = list(target.numpy())

            all_targ += target
            all_pred += pred

    print("evaluating network...")
    print(classification_report(all_targ, all_pred,
                                target_names=labelNames))


if __name__ == '__main__':
    parse_args()
    exp_path = args.exp
    writer = SummaryWriter(log_dir=exp_path)
    main()
