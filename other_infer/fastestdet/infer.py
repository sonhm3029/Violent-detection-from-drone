import os
import cv2
import onnx
import time
import argparse
from onnxsim import simplify

import torch
from utils.tool import *
from module.detector import Detector

if __name__ == '__main__':
    # 指定训练配置文件
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--yaml', type=str, default="", help='.yaml config')
    # parser.add_argument('--weight', type=str, default=None, help='.weight config')
    # parser.add_argument('--img', type=str, default='', help='The path of test image')
    # parser.add_argument('--thresh', type=float, default=0.65, help='The path of test image')
    # parser.add_argument('--onnx', action="store_true", default=False, help='Export onnx file')
    # parser.add_argument('--torchscript', action="store_true", default=False, help='Export torchscript file')
    # parser.add_argument('--cpu', action="store_true", default=False, help='Run on cpu')

    # opt = parser.parse_args()
    # assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
    # assert os.path.exists(opt.weight), "请指定正确的模型路径"
    # assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thresh = 0.2
        

    # 解析yaml配置文件
    cfg = LoadYaml("/home/hoang/Violence_detection_byDrone/FastestDet/configs/mydata.yaml")    
    print(cfg) 


    weight = "/home/hoang/Violence_detection_byDrone/FastestDet/checkpoint/best.pth"
    # 模型加载
    print("load weight from:%s"%weight)
    model = Detector(cfg.category_num, True).to(device)
    model.load_state_dict(torch.load(weight, map_location=device))
    #sets the module in eval node
    model.eval()
    
    video_path = "Violence_1_drone - 3of8.mp4"
    
    vid_capture = cv2.VideoCapture(video_path)

    while( vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            res_img = cv2.resize(frame, (cfg.input_width, cfg.input_height), interpolation = cv2.INTER_LINEAR) 
            img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
            img = img.to(device).float() / 255.0
            
            # Prediction
            preds = model(img)
            output = handle_preds(preds, device, thresh)
            
            
            LABEL_NAMES = ["violence", "non_violence"]
            
            H, W, _ = frame.shape
            scale_h, scale_w = H / cfg.input_height, W / cfg.input_width
            
            for box in output[0]:
                print(box)
                box = box.tolist()
       
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]

                x1, y1 = int(box[0] * W), int(box[1] * H)
                x2, y2 = int(box[2] * W), int(box[3] * H)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
                cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Frame", frame)
            cv2.waitKey(20)
        else:
            cv2.destroyAllWindows()
            break
    
