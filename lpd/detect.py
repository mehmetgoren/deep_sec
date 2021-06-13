import torch
import numpy as np
from lpd.layers.functions.prior_box import PriorBox
from lpd.utils.nms.py_cpu_nms import py_cpu_nms
from lpd.utils.box_utils import decode, decode_landm
import time
from lpd.data.config import cfg_mnet as cfg
from lpd.models.retina import Retina
import torch.backends.cudnn as cudnn
import argparse
import cv2
import pytesseract
from typing import List
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


parser = argparse.ArgumentParser(description='RetinaPL')

parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=1000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

args = parser.parse_args()

torch.set_grad_enabled(False)

# net and model
net = Retina(cfg=cfg, phase='test')
model_path ='../lpd/weights/mnet_plate.pth'
net = load_model(net, model_path, False)
net.eval()
print('Finished loading model!')
cudnn.benchmark = True
net = net.to(device)

resize = 1


# that's the detection method you are looking for
def detect_boxes(img_raw):
    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()

    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    box_list = list()
    dets = np.concatenate((dets, landms), axis=1)
    print('priorBox time: {:.4f}'.format(time.time() - tic))
    # show image
    if args.save_image:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            print(text)
            b = list(map(int, b))
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            box_list.append(np.array([x1, y1, x2, y2]))
    return box_list


def _get_pytesseract_result(test_license_plate):
    resize_test_license_plate = cv2.resize(
        test_license_plate, None, fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC)
    grayscale_resize_test_license_plate = cv2.cvtColor(
        resize_test_license_plate, cv2.COLOR_BGR2GRAY)
    gaussian_blur_license_plate = cv2.GaussianBlur(
        grayscale_resize_test_license_plate, (5, 5), 0)

    custom_config = r'--oem 3 -l eng --psm 6'  # '--oem 3 -l eng --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    new_predicted_result_GWT2180 = pytesseract.image_to_string(gaussian_blur_license_plate, lang='eng',
                                                               config=custom_config)
    filter_new_predicted_result_GWT2180 = "".join(new_predicted_result_GWT2180.split()).replace(":", "").replace(
        "-", "")
    # print(filter_new_predicted_result_GWT2180)
    return filter_new_predicted_result_GWT2180


class BoxLabelPair:
    def __init__(self, box, label):
        self.box = box
        self.label = label


def detect_boxes_and_labels(numpy_img) -> List[BoxLabelPair]:
    boxes_labels: List[BoxLabelPair] = []
    boxes = detect_boxes(numpy_img)
    for box in boxes:
        x1, y1, x2, y2 = box
        sub_img = numpy_img[y1:y2, x1:x2]
        label_pred = _get_pytesseract_result(sub_img)
        print('label_pred', label_pred)
        boxes_labels.append(BoxLabelPair(box, label_pred))
    return boxes_labels
