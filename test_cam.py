import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import cv2
import matplotlib.pyplot as plt
from utils.activation_map import feature_maps, normalize_output
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.general import check_file, set_logging, increment_path, check_dataset, check_img_size
from utils.torch_utils import select_device, time_synchronized
# from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

class _BaseCAM(object):
    def __init__(self, model):
        super(_BaseCAM, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = [] # a set of hook function handlers

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    def _encode_one_hot(self, idx):
        # when using YOLO network
        one_hot = torch.FloatTensor(1, self.preds.size()[-2], self.preds.size()[-1]).zero_()
        flag_idx = idx[0]
        one_hot[0][flag_idx] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()
        self.preds, _ = self.model(self.image)
        self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.sort(0, True)
        return self.prob, self.idx

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)

class gradCAM(_BaseCAM):
    def __init__(self, model, candidate_layers = None):
        super(gradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output[0].detach() #해당 모듈까지 텐서를 계산하고 그 이후에는 계산 안한다.

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():  # name = module 이름
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _normalize(self, grads):
        L2_NORMALIZATION = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / L2_NORMALIZATION

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def get_gradcam(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer) # feature map
        grads = self._find(self.grad_pool, target_layer) # gradient back prop
        weights = self._compute_grad_weights(grads)
        del grads

        grad_cam = (fmaps[0] * weights[0]).sum(dim = 0)
        del fmaps
        del weights

        grad_cam = torch.clamp(grad_cam, min=0.) # < - ReLU

        grad_cam -= grad_cam.min()
        grad_cam /= grad_cam.max()

        return grad_cam.detach().cpu().numpy()


def run_grad_cam():
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    target_layer = ['model.%s' % x for x in [9]]
    gradcam = gradCAM(model=model, candidate_layers=target_layer)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        probs, idx = gradcam.forward(img)
        # pred = model(img, augment=opt.augment)[0]

        # feature map Extract
        for target_layer in target_layer:
            gradcam.backward(idx=idx[0])
            output = gradcam.get_gradcam(target_layer=target_layer)

            p = Path(path)
            save_path = str(save_dir)  # img.png

            # save results (image with feature map)



def activation_map():
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt') # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)) # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)


    # Load model
    model = attempt_load(weights, map_location=device) # load FP32 model
    stride = int(model.stride.max()) # model stride
    imgsz = check_img_size(imgsz, s=stride) # check img_size

    # set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    target_layer = ['model.%s' % x for x in range(5)]
    f_maps = feature_maps(model=model, candidate_layers= None)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() # uint8 to fp16/32
        img /= 255.0 # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # feature map Extract
        for target_layer in target_layer:
            fmaps = f_maps.get_fmaps(target_layer=target_layer)
            fmaps_img = fmaps.cpu().numpy()
            fmaps_img = normalize_output(fmaps_img)
            fmaps_img = np.uint8(255*fmaps_img)
            p = Path(path)
            save_path = str(save_dir) # img.png

            # save results (image with feature map)
            if save_img:
                for i in range(fmaps_img.shape[0]):
                    heatmap = cv2.applyColorMap(fmaps_img[i], cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(save_path,'{}_{}.png'.format(target_layer,i)), np.uint8(heatmap))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/yolov5l_csp_upsample_sr_chx2/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='example', help='input image path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle componenet of cam_weights*activations')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/avtivation_map', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    # activation_map()
    with torch.autograd.set_detect_anomaly(True):
        run_grad_cam()
    # set_logging()
    # device = select_device(opt.device, batch_size=opt.batch_size)
    #
    # # Ditectories
    # save_dir = Path(increment_path(Path(opt.project)/ opt.name, exist_ok=opt.exist_ok)) # increment run
    #
    #
    #
    # # Load model
    # model = attempt_load(opt.weights, map_location= device) # load FP32 model
    # target_layer = ['model.15']
    #
    #
    # #load image
    # rgb_img = cv2.imread(opt.image_path, 1)[:,:,::-1]
    # rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
    #                                          std=[0.229, 0.224, 0.225])
    # # model output
    #
    # # get activation map
    # fig, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(input_tensor.detach().numpy)
    # # grad-CAM
    # grad_cam = GradCAM(model=model, target_layer=target_layer)
    # probs, idx = grad_cam.forward(input_tensor.to(device))
    # print(probs[2905], idx[0])
    # grad_cam.backward(idx=idx[0])
    #
    #
    # # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
    # # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    # # cv2.imwrite(f'grad_cam.jpg', cam_image)
