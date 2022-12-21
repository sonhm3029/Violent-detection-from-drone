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
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import datetime
import secrets
import string



NUM_FRAME_PER_SEQUENCE = 15
SOURCE_DATA = "/home/hoang/Violence_detection_byDrone/data"



def generateUniqueString():
    return ''.join(secrets.choice(string.ascii_uppercase + string.ascii_lowercase) for i in range(32))

def generateUniquePrefix():
    currentTime = datetime.datetime.now()
    prefixDate = "_".join(str(currentTime.date()).split("-"))
    prefixTimestamp = "".join(str(currentTime.timestamp()).split("."))
    return f"{prefixDate}_{prefixTimestamp}"

# def saveImgProcessing(listImgBbox):
#     """Find the top left most point 
#     and bottom right most point
#     """
#     # Find max number of bbox (events happen) in seq of images
#     totalEvents = max([len(bboxes) for bboxes in listImgBbox])
def roundZeros(coordinates: np.ndarray):
    result = [int(value) if value > 0 else 0 for value in coordinates]
    return result

def getCropBBox(listBboxes, padding=20):
    """
    Padding default 20px in width anh height
    """
    result = [roundZeros(np.array(xyxy.copy()) - padding) for xyxy in listBboxes]
    
    return result

def cropImg(img, topLeft, bottomRight):
    width = bottomRight[0] - topLeft[0]
    height = bottomRight[1] - topLeft[1]
    # cropped = img[start_row:end_row, start_col:end_col]
    return img[topLeft[1]: bottomRight[1], topLeft[0]: bottomRight[0]]

def detect():
    
    source, weights, imgsz = opt.source, opt.weights, opt.img_size

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    # set_logging()
    device = select_device(opt.device)
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
            tempImgsSeq = []
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
            pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=False)
        
        # Process detections
        # Save to csv obj
        # img, class, bbox list
        save_obj = [1,1, []]
        # Pred: [det]
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            save_obj[0] = im0
            # If detect obj in frame
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # List bounding boxes found in frame
                # xyxy -> top left (x, y) vs bottom right (x,y)
                save_obj[1] = 0
                save_obj[2] = [item[:4].tolist() for item in reversed(det)]
                    
                
            tempImgsSeq.append(tuple(save_obj))
            save_obj = [1,1, []]
        
        if count >= NUM_FRAME_PER_SEQUENCE + 1:
            tempImgsSeq.pop(0)
        
        # Check if 8th frame is having obj
        if count >= NUM_FRAME_PER_SEQUENCE and tempImgsSeq[7][1] == 0:
            listBboxes = getCropBBox(tempImgsSeq[7][2])
            for bbox_idx, bbox in  enumerate(listBboxes):
                folderName = generateUniqueString()
                folderDir = f"{SOURCE_DATA}/{folderName}"
                os.mkdir(folderDir)
                topLeft = bbox[:2]
                bottomRight = bbox[2:]
                for idx, save_img in enumerate(tempImgsSeq):
                    # Crop list image before save
                    print(f"Img {idx + 1} bbox number {bbox_idx + 2}:",bbox)
                    cv2.imwrite(f"{folderDir}/{folderName}_frame_{idx+1}.jpg",
                                cropImg(save_img[0], topLeft, bottomRight))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    print(opt)  
    
    with torch.no_grad():
        detect()    
