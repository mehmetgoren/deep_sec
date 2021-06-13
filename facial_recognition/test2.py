from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

print(help(InceptionResnetV1) )
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img_path = '/home/gokalp/Downloads'

# Get cropped and prewhitened image tensor
img = Image.open(img_path + '/beg.jpg')

# Calculate embedding (unsqueeze to add batch dimension)
img_cropped = mtcnn(img, save_path=img_path + '/cropped.jpg')

# Or, if using for VGGFace2 classification
resnet.classify = True
img_probs = resnet(img_cropped.unsqueeze(0))
x = img_probs.detach().numpy()
print(x)

