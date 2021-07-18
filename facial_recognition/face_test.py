from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

workers = 4
device = torch.device('cuda:0')

# those are default parameters
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
mtcnn.keep_all = True  # to detect all faces
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def collate_fn(x):
    return x[0]


dataset = datasets.ImageFolder(
    '/mnt/sdc1/facial_recognition/facenet_pytorch/data/test_images')
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
for x, y in loader:
    x_aligneds, probs = mtcnn(x, return_prob=True)
    if x_aligneds is not None:
        j = 0
        for x_aligned in x_aligneds:
            # yyy = t.view(1, t.shape[0],t.shape[1],t.shape[2])
            # xxx = yyy
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])
            print('Face detected with probability: ', probs[j])
            print(x_aligned)
            j += 1
        # print(prob)
        # print('Face detected with probability: {:8f}'.format(prob))
        # aligned.append(x_aligned)
        # names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
df = pd.DataFrame(dists, columns=names, index=names)
print(df)
del resnet
