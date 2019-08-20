import torch
import numpy as np
import cv2
import json
from models import VGGMiniCBR
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

video_path = '/home/mumu01/Projects/sample_video.mkv'
json_path = '/home/mumu01/Projects/proposals.json'

use_cuda = torch.cuda.is_available()

# fix random seeds
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
labelNames = ["top", "trouser", "pullover", "dress", "coat",
              "sandal", "shirt", "sneaker", "bag", "ankle boot"]

device = torch.device("cuda" if use_cuda else "cpu")


def get_prediction(model, img):
    # img to tensor
    model.eval()
    tr = transforms.Compose([transforms.CenterCrop((28, 28)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.2860,), (0.3530,))
                             ])
    img = Image.fromarray(img)

    ximg_tnsr = tr(img)

    ximg_tnsr = ximg_tnsr.to(device)
    ximg_tnsr = ximg_tnsr.unsqueeze(0)

    output = model(ximg_tnsr)
    output = F.log_softmax(output, dim=1)

    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    pred = pred.cpu().numpy().item()

    return pred, labelNames[pred]


with open(json_path) as handle:
    frame_objs_dict = json.loads(handle.read())

video = cv2.VideoCapture(video_path)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
print(width)
print(height)


def validate_bbox(bbox):
    for el in bbox:
        if el < 0:
            return False
    x1, y1, x2, y2 = bbox
    if x1 >= width or x2 >= width or y1 >= height or y2 >= height:
        return False
    return True


model = VGGMiniCBR(num_classes=10)
model.to(device)
fn = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # preprocessing
    gray_frame = gray_frame.astype(float)
    gray_frame /= 255.

    proposals_lst = frame_objs_dict[str(fn)]
    for obj_box in proposals_lst:
        x1, y1, x2, y2 = obj_box

        cv2.rectangle(frame, (x1, y1,), (x2, y2), (0, 255, 0), 2)
        # cv2.rectangle(frame, (y, h), (y,  h), (0, 255, 0), 2)
    # cv2.imshow("Show1", frame)
    # cv2.waitKey()
    c = 0
    for obj_box in proposals_lst:
        x1, y1, x2, y2 = obj_box
        if validate_bbox(obj_box):
            print("Box: ", obj_box)
            # obj = gray_frame[y1:y1 + y2, x1:x1 + x2]
            obj = gray_frame[y1:y2, x1: x2]
            # cv2.imshow("Show", obj)
            # cv2.waitKey()
            pred, cls_name = get_prediction(model, obj)
            c += 1
        else:
            print(obj_box)

    print(len(proposals_lst))
    print(c)
    exit()

    cv2.imshow("Show", frame)
    cv2.waitKey()
    exit()
    fn += 1
