import cv2
from detect import detect_draw, detect_crop
import time
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# burası plugin olarak tasarlaancak. Ormegin burda dahua dvr plug-in
img_path = '/media/gokalp/Super 2/Coding/ML/pytorch/deep_sec/resources/delete_later'
user = 'admin'
pwd = 'a12345678'
ip = '192.168.0.108'
port = '554'
camera = 2
subtype = 0  # 0 is main stream, 1 is extra stream
url = f'rtsp://{user}:{pwd}@{ip}:{port}/cam/realmonitor?channel={camera}&subtype={subtype}'


def capture_sec_cam():
    cap = cv2.VideoCapture(url)
    while 1:
        ret, frame = cap.read()
        detect_draw(frame)
        cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


def capture_sec_cam_with_fps(frame_rate=10):
    cap = cv2.VideoCapture(url)
    prev = 0
    while 1:  # (cap.isOpened()):
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if time_elapsed > 1. / frame_rate:
            prev = time.time()
            detect_draw(frame)
            cv2.imshow('VIDEO', frame)
            cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

# cuda_available = torch.cuda.is_available()
# class SSIM_PytorchImg:
#     def __init__(self, np_array):
#         self.img = np_array
#         # self.pred_cls = pred_cls
#     def to_tensor(self):
#         np_array = torch.from_numpy(self.img)
#         if cuda_available:
#             np_array = np_array.cuda()
#         return np_array
#     #buralar hep interface' e taşınacak
#
# def detected_before_pytorch(detected_imgs, img_numpy):
#     #convert to grayscale
#     img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
#     threshold = 1. / 5.
#     for detected_img in detected_imgs:
#         resized_img_numpy = cv2.resize(img_numpy, (detected_img.img.shape[1], detected_img.img.shape[0]))
#         torch_img_numpy = SSIM_PytorchImg(resized_img_numpy)
#         loss = pytorch_ssim.ssim(torch_img_numpy.to_tensor(), detected_img.to_tensor())
#         if loss > threshold:
#             return True
#     detected_imgs.append(SSIM_PytorchImg(img_numpy))
#     return False

def capture_sec_cam_with_save_image():
    cap = cv2.VideoCapture(url)
    detected_imgs = []

    def detected_before(img_numpy):
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)
        threshold = 1. / 5.
        for detected_img in detected_imgs:
            img_numpy = cv2.resize(img_numpy, (detected_img.shape[1], detected_img.shape[0]))
            loss = ssim(img_numpy, detected_img)
            if loss > threshold:
                return True
        detected_imgs.append(img_numpy)
        return False

    while 1:
        ret, frame = cap.read()
        infos = detect_crop(frame)
        if len(infos):
            for info in infos:
                img = info.img
                if not detected_before(img):
                    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                    full_file_name = f'{img_path}/{info.pred_cls}_{now}.jpg'
                    cv2.imwrite(full_file_name, img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

capture_sec_cam()