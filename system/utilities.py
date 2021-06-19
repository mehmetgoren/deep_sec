import torch
import matplotlib
from PIL import ImageColor
from typing import List, Tuple
import numpy as np
from datetime import datetime

device = torch.device('cuda:0')
half = device.type != 'cpu'

resources_path = '/mnt/37c00eec-d043-4088-b730-36c9e48a38e4/deep_sec/resources'
coco_names_path = resources_path + '/coco_names.txt'


class DetectedObject:
    def __init__(self):
        self.img: np.array = None
        self.text: str = None

    def get_text(self):
        return self.text

    def create_unique_key(self):
        return self.text

    def get_pred_color(self) -> Tuple[int, int, int]:
        return 0, 0, 0


class CocoInfo:
    def __init__(self):
        self.names = None
        self.colors = None

    def get_names(self) -> List[str]:
        if self.names is None:
            with open(coco_names_path) as f:
                self.names = f.read().splitlines()
        return self.names

    def get_name(self, index) -> str:
        return self.get_names()[index]

    def get_colors(self) -> List[Tuple[int, int, int]]:
        if self.colors is None:
            hex_list = list(matplotlib.colors.cnames.values())
            self.colors = [ImageColor.getrgb(hex) for hex in hex_list]
        return self.colors

    def get_color(self, index) -> Tuple[int, int, int]:
        return self.get_colors()[index]


_coco_info = CocoInfo()


class CocoDetectedObject(DetectedObject):
    def __init__(self):
        super(CocoDetectedObject, self).__init__()
        self.pred_score: float = None
        self.pred_cls_indx: int = None
        self.track_id = None

    def get_text(self):
        self.text = self.get_pred_cls() + (
            '_' + str(self.track_id) if self.track_id is not None else '') + ' ' + "{:.2f}".format(self.pred_score)
        return self.text

    def create_unique_key(self, detected: DetectedObject):
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        suffix = (str(self.track_id) if self.track_id is not None else '') + '_' + "{:.2f}".format(
            self.pred_score) + now
        key = f'{detected.pred_cls_indx}_{suffix}'
        return key

    def get_pred_cls(self) -> str:
        if self.pred_cls_indx is None:
            return ''
        return _coco_info.get_name(self.pred_cls_indx)

    def get_pred_color(self) -> Tuple[int, int, int]:
        if self.pred_cls_indx is None:
            return 0, 0, 0
        return _coco_info.get_color(self.pred_cls_indx)
