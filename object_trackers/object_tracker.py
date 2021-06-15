# Add as a submodule later: https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch
import numpy as np
# todo: please move all numpy codes to GPU by pytorch if you go to enterpirse for this feature
# todo: ek olarak cos similarty ile bulduğumuz class ve score' u deepsort ve tarcker' a parametreye taşıyarak gereksiz cos sim işleminden kurtul
from system.utilities import device, DetectedObject, CocoInfo
from typing import List
from .deep_sort.deep_sort import DeepSort
from .utils import parser
import torch
import torch.nn as nn
import cv2


coco_info = CocoInfo()

class ObjectTracker:
    def __init__(self):
        self.ds = self._get_deep_sort()

    @staticmethod
    def _bbox_rel(*xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = torch.min(xyxy[0], xyxy[2])
        bbox_top = torch.min(xyxy[1], xyxy[3])
        bbox_w = torch.abs(xyxy[0] - xyxy[2])
        bbox_h = torch.abs(xyxy[1] - xyxy[3])
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    @staticmethod
    def _get_deep_sort():
        cfg = parser.get_config()
        return DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    @staticmethod
    def _draw_detecteds(img, di, box) -> List[DetectedObject]:
        xy1 = tuple([box[0], box[1]])
        xy2 = tuple([box[2], box[3]])
        color = di.pred_color
        cv2.rectangle(img, xy1, xy2, color)
        text = ''
        text += di.pred_cls + ' ' + "{:.2f}".format(di.pred_score)
        if di.track_id is not None:
            text += '_' + str(di.track_id)
        cv2.putText(img, text, xy1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=1)
        return img

    def _find_class(self, boxes, output):
        """cpu version is:
        from scipy import spatial
        img1 = np.array([ 747.,   41., 1154.,  704.])
        img2 = np.array([ 745,   36, 1160,  710])
        result = 1. - spatial.distance.cosine(img1, img2)
        """
        for box in boxes:
            *xyxy, conf, cls = box
            img1 = torch.tensor(xyxy).to(device)
            img2 = torch.from_numpy(output).to(device)
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            result = cos(img1, img2)
            if result.item() > .98:
                return int(cls.item())
        return 0  # uhh, person?

    def _draw_boxes_not_mine(self, detected_boxes, img, bbox, pred_score, di, identities=None, offset=(0, 0)):
        def compute_color_for_labels(_label):
            """
            Simple function that adds fixed color depending on the class
            """
            palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
            _color = [int((p * (_label ** 2 - _label + 1)) % 255) for p in palette]
            return tuple(_color)

        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = compute_color_for_labels(id)
            # label = '{}{:d}'.format("", id) + f'_{cls}_{str(int(conf.item()))}'
            di.pred_cls_indx = self._find_class(detected_boxes, box)
            di.pred_cls = coco_info.get_name(di.pred_cls_indx)
            di.track_id = id
            print('id: ', id)
            label = f'{di.pred_cls} {pred_score:.2f} ({id:d})'
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img

    def detect(self, img: np.array, detected_boxes) -> List[DetectedObject]:
        dis = []
        # print('boxes length: ', len(detected_boxes))
        bbox_xywh = []
        confs = []
        # Adapt detections to deep sort input format
        for box in detected_boxes:
            *xyxy, conf, cls = box

            x_c, y_c, bbox_w, bbox_h = self._bbox_rel(*xyxy)
            obj = [x_c, y_c, bbox_w, bbox_h]
            bbox_xywh.append(obj)
            confs.append([conf])

            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            # Pass detections to deepsort
            outputs = self.ds.update(xywhs, confss, img)
            # print('outputs length: ', len(outputs))
            # print(outputs)
            # if len(dis):
            #     for di in dis:
            #         self._draw_detecteds(img, di, box)
            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                di = DetectedObject(img, conf, int(cls.item()))
                dis.append(di)
                # di.track_id = identities #should be removed if it is not indeed.
                self._draw_boxes_not_mine(detected_boxes, img, bbox_xyxy, conf, di, identities)
        return dis
