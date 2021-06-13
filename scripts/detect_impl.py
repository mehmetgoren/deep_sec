import PIL
import requests
import pafy
import numpy as np
import cv2
from detect import detect_draw

def detect_img():
    dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
    img_url = dir + 'zidane.jpg'
    img = np.asarray(PIL.Image.open(requests.get(img_url, stream=True).raw))
    img = detect_draw(img)
    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def capture_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, numpy_img = cam.read()
        if mirror:
            numpy_img = cv2.flip(numpy_img, 1)
        numpy_img = detect_draw(numpy_img)
        cv2.imshow('my webcam', numpy_img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()

def capture_yt_video():
    url = 'https://www.youtube.com/watch?v=RVWIr0-f7oU&t=4s' #"https://www.youtube.com/watch?v=wqctLW0Hb_0" 'https://www.youtube.com/watch?v=hBF2-hlvAQI' 'https://www.youtube.com/watch?v=BKXEaHFOeKc'

    video = pafy.new(url)
    streams = video.streams
    for i in streams:
        print(i)
    best = video.getbest()
    # start the video
    cap = cv2.VideoCapture(best.url)
    while True:
        ret, numpy_img = cap.read()
        detect_draw(numpy_img)
        cv2.imshow('yt', numpy_img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

capture_webcam()
capture_yt_video()