from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
from addict import Dict

workers = 4
device = torch.device('cuda:0')
img_dir = '/home/gokalp/Pictures/facial_recognition/train'


def collate_fn(x):
    return x[0]


dataset = datasets.ImageFolder(img_dir)
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
mtcnn = MTCNN(post_process=True, device=device)
mtcnn.keep_all = True  # to detect all faces
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        j = 0
        for t in x_aligned:
            aligned.append(t)
            names.append(dataset.idx_to_class[y])
            print('Face detected with probability: ', prob[j])
            j += 1
del mtcnn

aligned = torch.stack(aligned).to(device)
# those are default parameters
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
resnet.classify = True
embeddings = resnet(aligned).detach().cpu()

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
df = pd.DataFrame(dists, columns=names, index=names)
print(df)
del resnet

key = 0
dic = Dict()
classes = []
for name in names:
    if name in dic:
        classes.append(dic[name])
    else:
        dic[name] = key
        classes.append(key)
        key += 1

X = embeddings
y = np.array(classes)
# y = torch.from_numpy(y).to(device)
print('end')

# classification starts here. It may move to the softmax ann later.
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import pickle

svc = SVC(kernel="linear", probability=True)
svc.fit(X, y)

y_pred = svc.predict(X)
XXX = y_pred == y
acc = (y_pred == y).sum() / len(y) * 100.

print('y:      ', y)
print('y_pred: ', y_pred)
print('acc:    ', acc)

# lets evaluate the success rate.
cm = confusion_matrix(y_pred, y)
print(cm)
# save the model to disk
pickle.dump(svc, open('face_train_classifier_model.h5', 'wb'))

class_names = {v: k for k, v in dic.to_dict().items()}
pickle.dump(class_names, open('class_names.h5', 'wb'))

# dim = embeddings.shape[1]
# w = torch.autograd.Variable(torch.rand(dim), requires_grad=True)
# b = torch.autograd.Variable(torch.rand(1),   requires_grad=True)
# w, b = w.to(device), b.to(device)
#
# step_size = 1e-3
# num_epochs = 5000
# minibatch_size = 20
#
# for epoch in range(num_epochs):
#     inds = [i for i in range(len(X))]
#     for i in range(len(inds)):
#         L = max(0, 1 - y[inds[i]] * (torch.dot(w, X[inds[i]]) - b))**2
#         if L.item() != 0: # if the loss is zero, Pytorch leaves the variables as a float 0.0, so we can't call backward() on it
#             w.retain_grad()
#             b.retain_grad()
#             L.backward()
#             w.data -= step_size * w.grad.data # step
#             b.data -= step_size * b.grad.data # step
#             w.grad.data.zero_()
#             b.grad.data.zero_()
#
# def accuracy():
#     correct = 0
#     for i in range(len(y)):
#         y_predicted = int(np.sign((torch.dot(w, torch.Tensor(X[i])) - b).detach().numpy()[0]))
#         if y_predicted == y[i]: correct += 1
#     return float(correct)/len(y)
#
# print('accuracy', accuracy())
