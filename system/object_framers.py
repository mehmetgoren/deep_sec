from abc import ABC, abstractmethod
from typing import List
from system.object_detectors import ObjectDetectorBase, DetectedObject, CocoDetectedObject
from system.object_storages import ObjectStorageBase
import numpy as np
import cv2


class ObjectFramerBase(ABC):
    @abstractmethod
    def frame(self, object_detector: ObjectDetectorBase, storage: ObjectStorageBase, img: np.array) -> List[
        DetectedObject]:
        pass


class DrawObjectFramer(ObjectFramerBase):
    def frame(self, object_detector: ObjectDetectorBase, storage: ObjectStorageBase, img) -> List[DetectedObject]:
        dis = []
        boxes = object_detector.get_detect_boxes(img, storage)
        for i in range(len(boxes)):
            box = boxes[i]
            di: CocoDetectedObject = object_detector.create_detectedObject(box)
            di.img = img
            dis.append(di)
            color = di.get_pred_color()
            xy1 = tuple([box[0], box[1]])
            xy2 = tuple([box[2], box[3]])
            cv2.rectangle(img, xy1, xy2, color)
            text = di.get_text()
            cv2.putText(img, text, xy1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=1)
        return dis


class CropObjectFramer(ObjectFramerBase):
    def frame(self, object_detector: ObjectDetectorBase, storage: ObjectStorageBase, img) -> List[DetectedObject]:
        dis = []
        boxes = object_detector.get_detect_boxes(img, storage)
        for i in range(len(boxes)):
            box = boxes[i]
            y1, y2, x1, x2 = int(box[1]), int(box[3]), int(box[0]), int(box[2])
            sub_img = img[y1:y2, x1:x2]
            di = object_detector.create_detectedObject(box)
            di.img = sub_img
            dis.append(di)
        return dis
