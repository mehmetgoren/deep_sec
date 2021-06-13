from abc import ABC, abstractmethod
import cv2
from system.utilities import DetectedObject
from system.object_storages import DiskStorage


class ImageHandlerBase(ABC):
    @abstractmethod
    def handle(self, detected: DetectedObject):
        pass


class SaveImageHandler(ImageHandlerBase):
    # def __init__(self):
    #     self.img_path = resources_path + '/delete_later'
    #
    # def handle(self, detected: DetectedObject):
    #     key = self.create_unique_key(detected.pred_cls_indx, detected.pred_score)  # score may be tensor
    #     full_file_name = f'{self.img_path}/{key}.jpg'
    #     cv2.imwrite(full_file_name, detected.img)
    def __init__(self):
        self.storage = DiskStorage()

    def handle(self, detected: DetectedObject):
        self.storage.add(detected)


class ShowImageHandler(ImageHandlerBase):
    def __init__(self, wait_key=1, caption=False):
        self.wait_key = wait_key
        self.caption = caption

    def handle(self, detected: DetectedObject):
        cv2.imshow(detected.pred_cls if self.caption else 'window', detected.img)
        return cv2.waitKey(self.wait_key)
        # cv2.destroyAllWindows()
