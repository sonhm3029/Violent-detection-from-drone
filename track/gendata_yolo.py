import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


NUM_FRAME_PER_SEQUENCE = 15
SOURCE_DATA = "../../data"

def detect(opt):
    
    source, weights, imgsz = opt['source'], opt['weight'], opt['img-size']

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    # set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    count = 0
    tempVidCap = None
    frame_warm_up = int(NUM_FRAME_PER_SEQUENCE / 2)
    tempImgsSeq = []
    
    for img_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        print(path)
        """
            LOGIC TO EXTRACT FRAME
        """
        # Check if video change to make warm up
        if(vid_cap != tempVidCap):
            count = 0
            tempVidCap = vid_cap
        count +=1
        
        
        """
            INFER CODE
        """
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=False)
        
        # Process detections
        # Save to csv obj
        save_obj = [1,1]
        # Pred: [det]
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            save_obj[0] = im0
            # If detect obj in frame
            if len(det):
                save_obj[1] = 0
            tempImgsSeq.append(tuple(save_obj))
            save_obj = [1,1]
        
        if count >= NUM_FRAME_PER_SEQUENCE + 1:
            tempImgsSeq.pop(0)
        
        # Check if 8th frame is having obj
        if count >= NUM_FRAME_PER_SEQUENCE and tempImgsSeq[7][1] == 0:
            for img in tempImgsSeq:
                pass
