from system.stream_plugins import ImageUrlPlugin, ImageFolderPlugin
from system.object_framers import CropObjectFramer, DrawObjectFramer
from system.object_detectors import ObjectDetector, OnceDetector, TrackIdOnceDetector, SsimDetector
from system.image_handlers import ShowImageHandler, SaveImageHandler
from system.object_storages import InMemoryStorage
import torch
import time
import os


def test1():
    pl = ImageUrlPlugin('https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg')
    img = pl.get_img()

    detector = ObjectDetector()
    storage = InMemoryStorage()
    framer = CropObjectFramer()  # or DrawObjectFramer()
    handler = ShowImageHandler(0)  # or SaveImageHandler()

    infos = framer.frame(detector, storage, img)
    for info in infos:
        print(info.get_text())
        handler.handle(info)


def test2():
    pl = ImageUrlPlugin('https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg')
    img = pl.get_img()

    detector = OnceDetector()  # SsimDetector() # TrackIdOnceDetector()
    storage = InMemoryStorage()
    framer = CropObjectFramer()
    handler = SaveImageHandler()  # ShowImageHandler(0)

    while 1:
        infos = framer.frame(detector, storage, img)
        for info in infos:
            print(info.get_text())
            handler.handle(info)
        time.sleep(1)


# it is tracing from here
from object_trackers.object_tracker import ObjectTracker


def track_test():
    pl = ImageUrlPlugin('https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg')
    img = pl.get_img()

    tracker = ObjectTracker()
    detector = ObjectDetector()
    storage = InMemoryStorage()
    handler = SaveImageHandler()  # ShowImageHandler(1)
    while 1:
        boxes = detector.get_detect_boxes(img, storage)
        infos = tracker.detect(img, boxes)
        for info in infos:
            print(info.get_text())
            handler.handle(info)
        time.sleep(1)
    # wait_for_it = input()


# todo: bu kısım daha esnek bir detect info2 ya ihtiyaç duyuyor, onu sadece coco names ile sınırlı kılamayız
# todo: bunun için framer ve handler' ların modifiye olması gerel. Ayrıca lpd detector için de bu gerekli.
# todo: her şey commit olduktan sonra bununla ilgilendcek ve bu mesaj kaldırılacak
# todo: burada da artık framer ve handler kullanılıp, import düzenlenecek
def lpd_test():
    # todo: gerekli işlemlerden sonra burası kaldırılacak
    from lpd.detect import detect_boxes_and_labels
    import cv2
    test_img_dirs = '../lpd/imgs/turkish'
    img_dirs = os.path.expanduser(test_img_dirs)
    pl = ImageFolderPlugin(img_dirs)

    while not pl.closed():
        img = pl.get_img()
        infos = detect_boxes_and_labels(img)
        for info in infos:
            y1, y2, x1, x2 = int(info.box[1]), int(info.box[3]), int(info.box[0]), int(info.box[2])
            sub_img = img[y1:y2, x1:x2]
            cv2.imshow(info.label, sub_img)
            if cv2.waitKey(1000000) & 0xFF == ord('q'):
                continue


with torch.no_grad():
    test1()
    test2()
    track_test()
    lpd_test()
