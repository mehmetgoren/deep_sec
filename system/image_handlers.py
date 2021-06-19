from abc import ABC, abstractmethod
import cv2
from system.utilities import DetectedObject
from system.object_storages import DiskStorage


class ImageHandlerBase(ABC):
    @abstractmethod
    def handle(self, detected: DetectedObject):
        pass


class SaveImageHandler(ImageHandlerBase):
    def __init__(self):
        self.storage = DiskStorage()

    def handle(self, detected: DetectedObject):
        self.storage.add(detected)


class ShowImageHandler(ImageHandlerBase):
    def __init__(self, wait_key=1, caption=False):
        self.wait_key = wait_key
        self.caption = caption

    def handle(self, detected: DetectedObject):
        cv2.imshow(detected.get_text() if self.caption else 'window', detected.img)
        return cv2.waitKey(self.wait_key)
