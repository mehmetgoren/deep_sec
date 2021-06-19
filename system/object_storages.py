from abc import ABC, abstractmethod
from system.utilities import DetectedObject, resources_path
from addict import Dict
from redis import Redis
from typing import List
import cv2
import os
from pathlib import Path


class ObjectStorageBase(ABC):
    @abstractmethod
    def add(self, detected: DetectedObject):
        pass

    @abstractmethod
    def get_all(self) -> List[DetectedObject]:
        pass


class InMemoryStorage(ObjectStorageBase):
    def __init__(self):
        self.dic = Dict()

    def add(self, detected: DetectedObject):
        key = detected.create_unique_key(detected)
        print('dic count: ', len(self.dic))
        self.dic[key] = detected

    def get_all(self) -> List[str]:
        return self.dic.values()


class RedisStorage(ObjectStorageBase):
    def __init__(self):
        self.redis_client = Redis('127.0.0.1')

    # need to be serialized
    # def to_redis(r, a, n):
    #     """Store given Numpy array 'a' in Redis under key 'n'"""
    #     h, w = a.shape
    #     shape = struct.pack('>II', h, w)
    #     encoded = shape + a.tobytes()
    #
    #     # Store encoded data in Redis
    #     r.set(n, encoded)
    #     return
    #
    # def from_redis(r, n):
    #     """Retrieve Numpy array from Redis key 'n'"""
    #     encoded = r.get(n)
    #     h, w = struct.unpack('>II', encoded[:8])
    #     # Add slicing here, or else the array would differ from the original
    #     a = np.frombuffer(encoded[8:]).reshape(h, w)
    #     return a

    def add(self, detected: DetectedObject):
        key = detected.create_unique_key(detected)
        # img_to_bytes = cv2.imencode('.jpg', detected.img)[1].tobytes()
        # redis_cache.set('frame', frame_to_bytes)
        self.redis_client.set(key, detected)  # frame olmalı img

    # todo: not tested.
    def get_all(self) -> List[DetectedObject]:
        return self.redis_client.scan_iter(match='*')


# json' a gerek yok, resim adi, car_2021_06_16_1212 ve resimin kendisi yeterli
class DiskStorage(ObjectStorageBase):
    def __init__(self):
        self.file_extension = 'jpg'
        self.folder_name = f'{resources_path}/delete_later'

    def add(self, detected: DetectedObject):
        file_name = f'{self.folder_name}/{detected.create_unique_key(detected)}.{self.file_extension}'
        cv2.imwrite(file_name, detected.img)

    @staticmethod
    def _parse_file_name(file_name: str) -> DetectedObject:
        img = cv2.imread(file_name, 0)
        splits = file_name.split('_')
        cls_idx = splits[0]
        score = float(splits[1])
        ret = DetectedObject(img, score, cls_idx)
        return ret

    def get_all(self) -> List[DetectedObject]:
        ret: List[DetectedObject] = []
        for full_file_name in os.listdir():
            file_name = Path(full_file_name).stem
            detected = self._parse_file_name(file_name)
            ret.append(detected)
        return ret
