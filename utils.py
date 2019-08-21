import matplotlib

matplotlib.use("Agg")
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


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


def apply_dropout(m):
    if type(m) == nn.modules.dropout.Dropout2d:
        m.train()


def get_masked_pred(mean_pred, threshold):
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

    # for th in thresholds_arr:
    mask = np.where(entropy <= threshold, 1, 0)  # (10, 28, 28), 0, 0-> [BS, 1]

    # non_zeros = np.count_nonzero(mask)

    masked_pred = mask * maxed_pred
    # preds[str(th)] = masked_pred
    #
    # preds[str(th) + "_accepted"] = non_zeros

    return masked_pred, maxed_pred  # , non_zeros


def bb_intersection_over_union(boxA, boxB):
    # Funciton taken from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou