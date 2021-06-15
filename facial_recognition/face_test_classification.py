# Add as a submodule later: https://github.com/timesler/facenet-pytorch

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import pickle

workers = 16
device = torch.device('cuda:0')


def collate_fn(x):
    return x[0]

#You need to standartize face photo too small and too large picture are nat valid.
img_dir = '/home/gokalp/Pictures/facial_recognition/test'
dataset = datasets.ImageFolder(img_dir)
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
faces = []
with torch.no_grad():
    threshold = .83
    mtcnn = MTCNN(post_process=True, device=device)
    mtcnn.keep_all = True  # to detect all faces
    for x, y in loader:
        x_aligneds, probs = mtcnn(x, return_prob=True)
        if x_aligneds is not None:
            j = 0
            for x_aligned in x_aligneds:
                prob = probs[j]
                if prob < threshold:
                    continue
                aligned.append(x_aligned)
                print('Face detected with probability: ', prob)
                j += 1
            j = 0
            detected_faces, detected_faces_probs = mtcnn.detect(x)
            for detected_face in detected_faces:
                detected_faces_prob = detected_faces_probs[j]
                if detected_faces_prob < threshold:
                    continue
                print('face: ', detected_face, ', ', detected_faces_probs[j])
                face = x.crop((detected_face[0], detected_face[1], detected_face[2], detected_face[3]))
                faces.append(face)
                j += 1
    del mtcnn

with torch.no_grad():
    aligned = torch.stack(aligned).to(device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet.classify = True
    embeddings = resnet(aligned).detach().cpu()
    del resnet

svc = pickle.load(open('face_train_classifier_model.h5', 'rb'))
svc.probability = True
y_pred = svc.predict(embeddings)

probas_all = svc.predict_proba(embeddings)
probas = probas_all[:, 1]

class_names = pickle.load(open('class_names.h5', 'rb'))
index = 0
for p in y_pred:
    prob = probas[index]
    class_name = class_names[p] if prob <= .11 else 'other'
    face = faces[index]
    face.save(f'{img_dir}/{index}_{class_name}_{p}_{prob}.jpg')
    index += 1

print(y_pred)
print(class_names)
