from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import requests
import cv2
import pafy
import os


class SourcePluginBase(ABC):
    @abstractmethod
    def get_img(self) -> np.array:
        pass

    @abstractmethod
    def closed(self) -> bool:
        pass


class ImageUrlPlugin(SourcePluginBase):
    """img is numpy array"""

    def __init__(self, img_url):
        self.img_url = img_url

    def get_img(self) -> np.array:
        img = np.asarray(Image.open(requests.get(self.img_url, stream=True).raw))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def closed(self) -> bool:
        return True


class ImageFolderPlugin(SourcePluginBase):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.img_dirs = os.path.expanduser(folder_path)
        self.listdir = os.listdir(self.img_dirs)
        self.count = len(self.listdir)
        self.current_index = 0

    def get_img(self) -> np.array:
        if self.closed():
            return None
        current_file = os.path.join(self.img_dirs, self.listdir[self.current_index])
        img = cv2.imread(current_file)
        self.current_index += 1
        return img

    def closed(self) -> bool:
        return self.count <= self.current_index


class WebcamPlugin(SourcePluginBase):
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def get_img(self) -> np.array:
        ret_val, numpy_img = self.cam.read()
        return numpy_img

    def closed(self) -> bool:
        return not self.cam.isOpened()


class YoutubePlugin(SourcePluginBase):
    def __init__(self, url: str):
        self.video = pafy.new(url)
        best = self.video.getbest()
        self.cap = cv2.VideoCapture(best.url)

    def get_img(self) -> np.array:
        ret_val, numpy_img = self.cap.read()
        return numpy_img

    def closed(self) -> bool:
        return not self.cap.isOpened()


class DahuaDvrPlugin(SourcePluginBase):
    def __init__(self):
        # burası plugin olarak tasarlaancak. Ormegin burda dahua dvr plug-in
        # ayrıca bunu yaml' dan okuyalım
        user = 'admin'
        pwd = 'a12345678'
        ip = '192.168.0.108'
        port = '554'
        camera = 2
        subtype = 0  # 0 is main stream, 1 is extra stream
        url = f'rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={camera}&subtype={subtype}'
        self.cam = cv2.VideoCapture(url)

    def get_img(self) -> np.array:
        ret_val, numpy_img = self.cam.read()
        return numpy_img

    def closed(self) -> bool:
        return not self.cam.isOpened()
