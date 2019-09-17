import matplotlib

matplotlib.use('TkAgg')

import torch
import numpy as np
import cv2
import json
from models import VGGMiniCBR
from torchvision import transforms
import torchvision
from PIL import Image
import torch.nn.functional as F
import argparse
import models
from utils import apply_dropout, get_masked_pred, bb_intersection_over_union
import uuid
from matplotlib import pyplot as plt

args = None


def parse_args():
    global args
    parser = argparse.ArgumentParser(description='Generating predictions on input video')

    parser.add_argument('--checkpoint', required=True, type=str, help='Path to trained model.')
    parser.add_argument('--model', type=str, choices=['VGGMiniCBR', 'VGGMini'],
                        default='VGGMiniCBR', help='Choice of model (default: VGGMiniCBR)')
    parser.add_argument('--video', type=str, default='/home/mumu01/Projects/sample_video.mkv',
                        help='path to input video')
    parser.add_argument('--proposals', type=str, default='/home/mumu01/Projects/proposals.json',
                        help='path to JSON file which contains proposal Bounding boxes for each frame of the corresponding video.')
    parser.add_argument('--bayes', type=int, default=0,
                        help='To use MCDropout with args.bayes samples. default(0 for not using)')

    parser.add_argument('--exp', type=str, default='./exps/',
                        help='path to exp folder (default: ./exps/)')
    parser.add_argument('--percentile', type=float, default=0.25,
                        help='When using MCDropout, percentile threshold is used to ignore predicitons where entropy is higher than a particular threshold. (default 0.25)')
    args = parser.parse_args()


def validate_bbox(bbox):
    for el in bbox:
        if el < 0:
            return False
    x1, y1, x2, y2 = bbox
    if x1 >= width or x2 >= width or y1 >= height or y2 >= height:
        return False
    return True


parse_args()
video_path = args.video
json_path = args.proposals

use_cuda = torch.cuda.is_available()

max_entropy = torch.log(torch.tensor(10.0))
percentile = args.percentile
threshold = percentile * max_entropy
threshold = threshold.data.numpy()

# fix random seeds
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]
colrs = [(111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 30),
         (250, 170, 160), (102, 102, 156), (190, 153, 153), (150, 100, 100), (0, 80, 100)]
device = torch.device("cuda" if use_cuda else "cpu")


def get_prediction_bayes(model, img):
    model.eval()
    # print('Applying MC Dropout')
    model.apply(apply_dropout)

    n_samples = args.bayes
    # print('n_samples : {}'.format(n_samples))

    tr = transforms.Compose([
        transforms.Resize(28),
        transforms.RandomCrop((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    img = Image.fromarray(img)

    ximg_tnsr = tr(img)

    ximg_tnsr = ximg_tnsr.to(device)
    ximg_tnsr = ximg_tnsr.unsqueeze(0)

    model_out = None
    for q in range(n_samples):
        with torch.no_grad():
            if model_out is None:
                model_out = model(ximg_tnsr).detach().data
            else:
                model_out = model_out + model(ximg_tnsr).detach().data

    model_out = model_out / n_samples

    # masking predictions where entropy is high
    model_out = model_out.cpu()  # [1000, 10] --> [BS, C] -> shape (B, C, W, H)
    model_out = model_out.unsqueeze(2).unsqueeze(3)

    masked_pred, maxed_pred = get_masked_pred(model_out, threshold)

    masked_pred = masked_pred.squeeze()  # (1000, )
    maxed_pred = maxed_pred.squeeze()  # (1000, )
    if not masked_pred == maxed_pred:  # means that it is dropped
        return -1, -1

    else:
        return masked_pred, labelNames[masked_pred]


def get_prediction(model, img):
    # img to tensor
    model.eval()
    tr = transforms.Compose([
        transforms.Resize(28),
        transforms.RandomCrop((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    img = Image.fromarray(img)

    ximg_tnsr = tr(img)

    ximg_tnsr = ximg_tnsr.to(device)
    ximg_tnsr = ximg_tnsr.unsqueeze(0)


    # unique_filename = "./outs/" + str(uuid.uuid4()) + ".png"
    # torchvision.utils.save_image(ximg_tnsr, unique_filename)
    # return 0, '1'

    output = model(ximg_tnsr)
    output = F.log_softmax(output, dim=1)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    pred = pred.cpu().numpy().item()

    return pred, labelNames[pred]


def remove_overlaps(proposals):
    for obj1 in proposals:
        for obj2 in proposals:
            iou = bb_intersection_over_union(obj1, obj2)
            if iou > 0.1:
                proposals.remove(obj2)

    return proposals


with open(json_path) as handle:
    frame_objs_dict = json.loads(handle.read())

video = cv2.VideoCapture(video_path)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
print(width)
print(height)

model = models.__dict__[args.model](num_classes=10)
model.load_state_dict(torch.load(args.checkpoint))
print("Model: " + str(args.checkpoint) + " loaded successfully.")
model.to(device)
fn = 0

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
txt = "nobayes"
if args.bayes:
    txt = "bayes" + str(args.bayes)
out = cv2.VideoWriter('./output-' + txt + '.avi', fourcc, 30, (int(width / 2), int(height / 2)))
if args.bayes:
    print("Using MC Dropout.")
while True:
    ret, frame = video.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # preprocessing
    gray_frame = gray_frame.astype(float)
    gray_frame /= 255.

    proposals_lst = frame_objs_dict[str(fn)]
    proposals_lst = remove_overlaps(proposals_lst)

    c = 0
    for obj_box in proposals_lst:
        x1, y1, x2, y2 = obj_box
        if validate_bbox(obj_box):
            obj = gray_frame[y1:y2, x1: x2]
            obj = 1 - obj
            # obj *= 255
            # obj = obj.astype(np.uint8)
            #
            # plt.hist(obj.ravel(), 256, [0, 256])
            # plt.savefig('from_video2.png')
            # exit()
            if args.bayes:
                pred, cls_name = get_prediction_bayes(model, obj)
            else:
                pred, cls_name = get_prediction(model, obj)
            if pred >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), colrs[pred], 2)
                cv2.putText(img=frame, text=cls_name, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=colrs[pred], thickness=2)
            c += 1

    frame = cv2.resize(frame, dsize=(int(width / 2), int(height / 2)))
    out.write(frame)

    fn += 1
out.release()
video.release()
