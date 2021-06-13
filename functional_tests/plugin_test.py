import torch
import time
from system.stream_plugins import WebcamPlugin, YoutubePlugin, DahuaDvrPlugin
from system.object_framers import CropObjectFramer, DrawObjectFramer
from system.object_detectors import ObjectDetector, OnceDetector, TrackIdDetector, SsimDetector
from system.image_handlers import ShowImageHandler, SaveImageHandler
from system.object_storages import InMemoryStorage
from system.utilities import DetectedObject

url = 'https://www.youtube.com/watch?v=bBj7QqmFNPw'  # 'https://www.youtube.com/watch?v=OwAQB52Hv4M' # 'https://www.youtube.com/watch?v=bBj7QqmFNPw' #'https://www.youtube.com/watch?v=sPVlw5Zhr7k'
plugin = YoutubePlugin(url)  # DahuaDvrPlugin()  # YoutubePlugin(url)  # WebcamPlugin()

def capture_show(frame_rate=30):
    detector = ObjectDetector()
    storage = InMemoryStorage()
    framer = DrawObjectFramer()  # DrawObjectFramer() CropObjectFramer()
    handler = ShowImageHandler(1)  # or SaveImageHandler()
    prev = 0

    while not plugin.closed():
        time_elapsed = time.time() - prev
        img = plugin.get_img()
        if time_elapsed > 1. / frame_rate:
            prev = time.time()
            infos = framer.frame(detector, storage, img)
            if len(infos):
                for info in infos:
                    print(f'{info.pred_cls} - {info.pred_score}')
                    handler.handle(info)
            else:  # show even nothing is detected
                o = DetectedObject(img, 1., 1)
                handler.handle(o)


def capture_save():
    detector = OnceDetector()
    storage = InMemoryStorage()
    framer = CropObjectFramer()
    handler = SaveImageHandler()

    while not plugin.closed():
        img = plugin.get_img()
        infos = framer.frame(detector, storage, img)
        if len(infos):
            for info in infos:
                print(f'{info.pred_cls} - {info.pred_score}')
                handler.handle(info)


with torch.no_grad():
    capture_show()
    capture_save()

# it is tracing from here
from object_trackers.object_tracker import ObjectTracker


def track_test():
    tracker = ObjectTracker()
    detector = ObjectDetector()
    storage = InMemoryStorage()
    handler = ShowImageHandler(1)

    while not plugin.closed():
        img = plugin.get_img()
        boxes = detector.get_detect_boxes(img, storage)
        infos = tracker.detect(img, boxes)
        if len(infos):
            for info in infos:
                print(f'{info.pred_cls} - {info.pred_score} - {info.track_id}')
                handler.handle(info)
        else:  # show even nothing is detected
            o = DetectedObject(img, 1., 1)
            handler.handle(o)


with torch.no_grad():
    track_test()
