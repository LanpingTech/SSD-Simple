import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from config import parse_args
from voc_dataset import VOC_LABELS
from model import SSDPlus
from utils import MultiBoxEncoder, preproc_for_test, detect

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
ckpt_path = 'weights/90-loss-74.58.pth'
img_path = 'dataset/JPEGImages/val_044.jpg'

args = parse_args()

model = SSDPlus(args.num_classes, args.anchor_num, pretrain=True).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

mbe = MultiBoxEncoder(args)

ori_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
image = preproc_for_test(ori_image, args.min_size, args.mean)
image = torch.from_numpy(image)

with torch.no_grad():
    loc, conf = model(image.unsqueeze(0).to(device))
    loc = loc[0]
    conf = conf[0]
    conf = F.softmax(conf, dim=1).cpu()
    conf = conf.numpy()
    loc = loc.cpu().numpy()

decode_loc = mbe.decode(loc)
gt_boxes, gt_confs, gt_labels = detect(decode_loc, conf, nms_threshold=0.45, gt_threshold=0.1)
h, w = ori_image.shape[:2]
gt_boxes[:, 0] = gt_boxes[:, 0] * w
gt_boxes[:, 1] = gt_boxes[:, 1] * h
gt_boxes[:, 2] = gt_boxes[:, 2] * w
gt_boxes[:, 3] = gt_boxes[:, 3] * h
gt_boxes = gt_boxes.astype(int)


plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.imshow(ori_image[:, :, ::-1])
for box, label, score in zip(gt_boxes, gt_labels, gt_confs):
        print(label, score, *box)
        if score > 0.15:
            text = '{:s}:{:.2f}'.format(VOC_LABELS[label], score)
            ax.add_patch(
                plt.Rectangle(
                (box[0], box[1]), 
                 box[2] - box[0], 
                 box[3] - box[1],
                 fill=False, 
                 edgecolor='red',
                 linewidth=2)
            )
            ax.text(box[0], box[1], text,
                    bbox={'facecolor' : 'white', 'alpha' : 0.7, 'pad' : 5}
                   )
plt.savefig('pred.jpg')




