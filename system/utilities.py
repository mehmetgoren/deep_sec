import torch
import matplotlib
from PIL import ImageColor
from typing import List, Tuple

device = torch.device('cuda:0')
half = device.type != 'cpu'

resources_path = '/mnt/37c00eec-d043-4088-b730-36c9e48a38e4/deep_sec/resources'
coco_names_path = resources_path + '/coco_names.txt'


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


class DetectedObject:
    def __init__(self, img, pred_score: float, pred_cls_indx: int):
        self.img = img
        self.pred_score = pred_score
        self.pred_cls_indx = pred_cls_indx
        self.pred_cls = _coco_info.get_name(pred_cls_indx)
        self.pred_color = _coco_info.get_color(pred_cls_indx)
        self.track_id = None
