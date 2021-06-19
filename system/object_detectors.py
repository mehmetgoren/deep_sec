from system.object_storages import ObjectStorageBase
from system.utilities import device, half, DetectedObject, CocoDetectedObject
from abc import ABC, abstractmethod
from object_trackers.object_tracker import ObjectTracker
import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage.metrics import structural_similarity as sk_ssim
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent.futures
from datetime import datetime
from addict import Dict


class ObjectDetectorBase(ABC):
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(device)
        if half:
            self.model.half()  # to FP16

    def _get_detect_boxes(self, img: np.array):
        # activate later. Maybe we can get more accuracy ?
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.model(img)
        # print img1 predictions (pixels)
        #                   x1           y1           x2           y2   confidence        class
        # tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
        #         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
        #         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])
        # Output will be a numpy array in the following format:
        # [[x1, y1, x2, y2, confidence, class]]
        boxes = detections.xyxy[0]
        return boxes

    @abstractmethod
    def get_detect_boxes(self, img, storage: ObjectStorageBase):
        pass

    @abstractmethod
    def create_detectedObject(self, box) -> CocoDetectedObject:
        pass


def _create_coco_detected_object(box) -> CocoDetectedObject:
    x1, y1, x2, y2, confidence, cls = box
    obj = CocoDetectedObject()
    obj.pred_score = confidence.item()
    obj.pred_cls_indx = int(cls.item())
    return obj


class ObjectDetector(ObjectDetectorBase):
    def get_detect_boxes(self, img: np.array, storage: ObjectStorageBase):
        return self._get_detect_boxes(img)

    def create_detectedObject(self, box) -> CocoDetectedObject:
        return _create_coco_detected_object(box)


# todo: add white list for coco object types.
class OnceDetector(ObjectDetectorBase):
    def __init__(self):
        super(OnceDetector, self).__init__()
        self.device = torch.device('cuda:0')
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def _cosine_similarity(self, img1, img2) -> float:
        # cpu version:
        # from scipy import spatial
        # img1 = np.array([ 747.,   41., 1154.,  704.])
        # img2 = np.array([ 745,   36, 1160,  710])
        # result = 1. - spatial.distance.cosine(img1, img2)
        # return result

        # GPU version:
        # img1 = torch.tensor(img1).to(self.device)
        # img2 = torch.from_numpy(img2).to(self.device)
        result = self.cos(img1, img2)
        return result.item()

    def get_detect_boxes(self, img: np.array, storage: ObjectStorageBase):
        boxes = self._get_detect_boxes(img)
        arr = []
        detected_list = storage.get_all()

        for box in boxes:
            img1, conf, cls = box[0:4], box[4], box[5]
            cls_idx = int(cls.item())
            already_detected = False
            for detected in detected_list:
                if detected.pred_cls_indx != cls_idx:
                    continue
                # todo: you may open it later if it increase the accuracy
                # elif abs(detected.pred_score.item() - conf.item()) > .1:
                #     continue
                img2 = detected.img  # it stores DetectedImage
                similarity = self._cosine_similarity(img1, img2)
                if similarity > .98:  # it' s a threshold
                    print(f'Already detected {cls}-{conf} at {datetime.now()}')
                    already_detected = True
                    break  # already detected
            # if it is not detected before
            if not already_detected:
                arr.append(box)
                do = self.create_detectedObject(box)
                do.img = img1
                do.pred_score = conf
                do.pred_cls_indx = cls_idx
                storage.add(do)
        if not len(arr):
            return torch.empty(0)
        ret = torch.zeros((len(arr), 6))
        for j, item in enumerate(arr):
            ret[j] = item
        return ret

    def create_detectedObject(self, box) -> CocoDetectedObject:
        return _create_coco_detected_object(box)


class TrackIdOnceDetector(ObjectDetectorBase):
    def __init__(self):
        super(TrackIdOnceDetector, self).__init__()
        self.hash = set()
        self.tracker = ObjectTracker()

    def get_detect_boxes(self, img, storage: ObjectStorageBase):
        detected_boxes = self._get_detect_boxes(img)
        detecteds = self.tracker.detect(img, detected_boxes)
        arr = []
        for j, detected in enumerate(detecteds):
            if detected.track_id in self.hash:  # or detected.track_id is None
                print(self.hash)
                print(
                    f'Already detected {detected.track_id} {detected.pred_cls}-{detected.pred_score} at {datetime.now()}')
                continue
            arr.append(detected_boxes[j])
            print(f'detected {detected.track_id} {detected.pred_cls}-{detected.pred_score} at {datetime.now()}')
            self.hash.add(detected.track_id)
        if not len(arr):
            return torch.empty(0)
        ret = torch.zeros((len(arr), 6))
        for j, item in enumerate(arr):
            ret[j] = item
        return ret

    def create_detectedObject(self, box) -> CocoDetectedObject:
        return _create_coco_detected_object(box)


# todo: bir TrackIdDetector a daha ihtiyac var, once olmayan


# todo: implement it with redis later.
class SsimDetector(ObjectDetectorBase):
    def __init__(self):
        super(SsimDetector, self).__init__()
        # self.redis_client = Redis('127.0.0.1')
        self.dic = Dict()

    def get_detect_boxes(self, img, storage: ObjectStorageBase):
        detected_boxes = self._get_detect_boxes(img)
        arr = []
        for i in range(len(detected_boxes)):
            box = detected_boxes[i]
            y1, y2, x1, x2 = int(box[1]), int(box[3]), int(box[0]), int(box[2])
            sub_img = img[y1:y2, x1:x2]
            pred_score = box[4]
            cls_index = int(box[5])
            di = _create_coco_detected_object(box)
            di.img = sub_img
            if not self._detected_before(di, self.dic):
                arr.append(box)
        if not len(arr):
            return torch.empty(0)
        ret = torch.zeros((len(arr), 6))
        for j, item in enumerate(arr):
            ret[j] = item
        return ret

    def create_detectedObject(self, box) -> DetectedObject:
        return _create_coco_detected_object(box)

    @staticmethod
    def _ssim(img1, img2, make_it_grayscale=False) -> float:
        if make_it_grayscale:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        loss = sk_ssim(img1, img2)
        return loss

    def _detected_before_mt(self, detected: CocoDetectedObject, detected_dic: Dict, threshold=1. / 5.) -> bool:
        # make it grayscale before processing
        img = cv2.cvtColor(detected.img, cv2.COLOR_BGR2GRAY)
        if not detected.pred_cls in detected_dic:
            detected_dic[detected.pred_cls] = []
            detected_dic[detected.pred_cls].append(img)
            return False
        detected_imgs = detected_dic[detected.pred_cls]
        if not len(detected_imgs):
            detected_imgs.append(img)
            return False

        def fn_detected_before(img1, img2, threshold_):
            loss = self._ssim(img1, img2)
            return loss > threshold_

        max_workers = len(detected_imgs)
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for item in detected_imgs:
                futures.append(executor.submit(fn_detected_before, img, item, threshold))
        # for more informations: https://stackoverflow.com/questions/52082665/store-results-threadpoolexecutor
        futures, _ = concurrent.futures.wait(futures)
        result = False
        for future in futures:
            result = result or future.result()
        if not result:
            detected_imgs.append(img)
        return result

    def _detected_before(self, detected: CocoDetectedObject, detected_dic, threshold=1. / 5.) -> bool:
        # make it grayscale before processing
        img = cv2.cvtColor(detected.img, cv2.COLOR_BGR2GRAY)
        if not detected.pred_cls in detected_dic:
            detected_dic[detected.pred_cls] = []
            detected_dic[detected.pred_cls].append(img)
            return False
        detected_imgs = detected_dic[detected.pred_cls]
        if not len(detected_imgs):
            detected_imgs.append(img)
            return False
        for detected_img in detected_imgs:
            loss = self._ssim(img, detected_img)
            if loss > threshold:
                return True
        return False
