import torch
import torchvision
import numpy as np


def zca(x):
    """Computes ZCA transformation for the dataset.

    Args:
        x: dataset.
    Returns:
        ZCA transformation matrix and mean matrix.
    """
    [B, C, H, W] = list(x.size())
    x = x.reshape((B, C * H * W))  # flattern the data
    mean = torch.mean(x, dim=0, keepdim=True)
    x -= mean
    covariance = torch.matmul(x.transpose(0, 1), x) / B
    U, S, V = np.linalg.svd(covariance.numpy())
    eps = 1e-3
    W = np.matmul(np.matmul(U, np.diag(1. / np.sqrt(S + eps))), U.T)
    return torch.tensor(W), mean


def save_zca():
    print("Saving ZCA file.")
    # whiten FashionMNIST
    trainset = torchvision.datasets.FashionMNIST(
        root='data', transform=torchvision.transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=60000, shuffle=False, num_workers=4)
    for _, data in enumerate(trainloader):
        break
    images, _ = data
    W, mean = zca(images)
    torch.save(W, './statistics/fashionmnist_zca_3.pt')
