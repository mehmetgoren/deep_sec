import torch
import numpy as np
import cv2
import matplotlib
from PIL import ImageColor
from typing import List

with open('/object_detectors/coco_names.txt') as f:
    coco_names = f.read().splitlines()

# coco_colors = []
# for j in range(len(coco_names)):
#     coco_colors.append(list(np.random.random(size=3) * 256))
#     time.sleep(0)
hex_list = list(matplotlib.colors.cnames.values())
coco_colors = [ImageColor.getrgb(hex) for hex in hex_list]

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def get_detect_boxes(img_array):
    results = model(img_array)
    boxes = results.xyxy[0]
    # print img1 predictions (pixels)
    #                   x1           y1           x2           y2   confidence        class
    # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
    #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
    #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])
    # Output will be a numpy array in the following format:
    # [[x1, y1, x2, y2, confidence, class]]
    return boxes


def detect_draw(img_array) -> np.array:
    boxes = get_detect_boxes(img_array)
    for i in range(len(boxes)):
        box = boxes[i]
        xy1 = tuple([box[0], box[1]])
        xy2 = tuple([box[2], box[3]])
        pred_score = box[4]
        cls_index = int(box[5])
        pred_cls = coco_names[cls_index]
        color = coco_colors[cls_index]
        cv2.rectangle(img_array, xy1, xy2, color)
        text = pred_cls + ' ' + "{:.2f}".format(pred_score)
        cv2.putText(img_array, text, xy1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=1)

    return img_array


class CropImgInfo:
    def __init__(self, img, pred_score, pred_cls):
        self.img = img
        self.pred_score = pred_score
        self.pred_cls = pred_cls


def detect_crop(img_array) -> List[CropImgInfo]:
    crop_imgs = []
    boxes = get_detect_boxes(img_array)
    for i in range(len(boxes)):
        box = boxes[i]
        y1, y2, x1, x2 = int(box[1]), int(box[3]), int(box[0]), int(box[2])
        sub_img = img_array[y1:y2, x1:x2]
        pred_score = box[4]
        cls_index = int(box[5])
        pred_cls = coco_names[cls_index]
        crop_img = CropImgInfo(sub_img, pred_score, pred_cls)
        crop_imgs.append(crop_img)

    return crop_imgs
