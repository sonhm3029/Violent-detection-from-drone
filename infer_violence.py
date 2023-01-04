import argparse
import time
from pathlib import Path
import os
import numpy as np

import cv2
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import datetime
import secrets
import string

def detect():
    
    weights = "/home/hoang/Violence_detection_byDrone/yolov7/runs/train/exp22/weights/best.pt"
    imgsz = 640
    conf_thres = 0.8
    iou_thres = 0.65
    view_img = True

    device = select_device("cuda:0"if torch.cuda.is_available() else "cpu")
    half = device.type != 'cpu'  # half precision only supported on CUDA


    set_logging()   
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16
    # Set Dataloader
    vid_path, vid_writer = None, None

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    video_path = "Violence_1_drone - 3of8.mp4"
    
    vid_capture = cv2.VideoCapture(video_path)

    while( vid_capture.isOpened()):
        ret, img = vid_capture.read()
        im0s = np.copy(img)
        if ret == True:
            # img = cv2.resize(img, (imgsz, imgsz), interpolation = cv2.INTER_LINEAR) 
            # img = img.reshape(1, imgsz, imgsz, 3)
            img = letterbox(img, imgsz, stride=stride)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
                
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=False)[0]
        
            pred = model(img, augment=False)[0]
            
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=None)
            
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0 = '', im0s

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Stream results
                cv2.imshow("Frame", im0)
                cv2.waitKey(1)  # 1 millisecond
            
        else:
            cv2.destroyAllWindows()
            break
    
if __name__ == "__main__":
    detect()