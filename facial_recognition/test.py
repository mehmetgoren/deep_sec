from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import torch
import pafy

# Create face detector
device = torch.device('cuda:0')
mtcnn = MTCNN(select_largest=False, device=device)

# url = 'https://www.youtube.com/watch?v=bBj7QqmFNPw'
# video = pafy.new(url)
# streams = video.streams
# for i in streams:
#     print(i)
# best = video.getbest()
# cap = cv2.VideoCapture(best.url)
cap = cv2.VideoCapture()


# cap = cv2.VideoCapture(0)
def show_img(numpy_img):
    cv2.imshow('my webcam', numpy_img)
    cv2.waitKey(1)


with torch.no_grad():
    while True:
        ret_val, numpy_img = cap.read()
        frame = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        faces = mtcnn.detect(frame)
        if faces[0] is None:
            show_img(numpy_img)
            continue
        for boxes in faces:
            if len(boxes.shape) < 2:
                show_img(numpy_img)
                continue
            for box in boxes:
                if box.shape[0] < 4:
                    show_img(numpy_img)
                    continue
                xy1 = tuple([box[0], box[1]])
                xy2 = tuple([box[2], box[3]])
                cv2.rectangle(numpy_img, xy1, xy2, (255, 176, 116))
            show_img(numpy_img)
cap.release()
cv2.destroyAllWindows()

# Load a single image and display
v_cap = cv2.VideoCapture(
    '/mnt/sdc1/facial_recognition/facenet_pytorch/examples/video.mp4')
success, cv2_img = v_cap.read()
frame = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
frame = Image.fromarray(frame)

plt.figure(figsize=(12, 8))
plt.imshow(frame)
plt.axis('off')
plt.show()

# Detect face
faces = mtcnn.detect(frame)
with torch.no_grad():
    for boxes in faces:
        for box in boxes:
            xy1 = tuple([box[0], box[1]])
            xy2 = tuple([box[2], box[3]])
            cv2.rectangle(cv2_img, xy1, xy2, (255, 176, 116))
        cv2.imshow('face', cv2_img)
        cv2.waitKey()
# faces = mtcnn(frame) #.detect(frame) #or mtcnn(frame)
# with torch.no_grad():
#     for torch_img in faces:
#         img = torch_img.numpy()
#         cv2.imshow('face', img)
#         cv2.waitKey()
# narr = face.detach().numpy()
# print(face.shape)
del mtcnn
