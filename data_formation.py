import torch
from torchvision import datasets, transforms
import os
import cv2


def get_num_files(direc):
    return len([name for name in os.listdir(direc) if os.path.isfile(direc+'/'+name)])


labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle_boot"]

split = 'train'
root = './data/FashionMNIST/original/' + split

if not os.path.exists(root):
    os.makedirs(root)
for cls in labelNames:
    if not os.path.exists(root + '/' + cls):
        os.makedirs(root + '/' + cls)

train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data/', train=split == 'train', download=True,
                          transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=5000, shuffle=False, drop_last=False)

for data, target in train_loader:
    for img, lbl in zip(data, target):
        img = img.numpy().squeeze() * 255

        cls_name = labelNames[lbl]
        cls_path = root + '/'+cls_name
        comp_path = cls_path + '/'+str(get_num_files(cls_path)) + ".png"

        cv2.imwrite(comp_path, img)
