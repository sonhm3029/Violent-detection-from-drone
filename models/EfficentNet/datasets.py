import torch
from torchvision.models import efficientnet_b0
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from tqdm import tqdm
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt

pretrain_model = efficientnet_b0(pretrain=True)


DEFAULT_DATA_PATH ="data"
NUM_IMG_PER_SEQ = 15

def splitDataset(train_ratio, dataset_root = DEFAULT_DATA_PATH):
    
    if(os.path.exists(f"{dataset_root}/dataset_info.csv")):
        os.remove(f"{dataset_root}/dataset_info.csv")
    
    listDataFolder = os.listdir(dataset_root).copy()
    list_data = []
    
    total_len = len(listDataFolder)
    train_len = int(train_ratio * total_len)
    
    for idx, folderName in enumerate(listDataFolder):
        
        if(len(os.listdir(f"{dataset_root}/{folderName}")) <NUM_IMG_PER_SEQ):
            print(f"Found empty or not enough images in seq folder, remove folder {folderName}")
            # os.remove(f"{dataset_root}/{folderName}")
            continue
        
        if (idx + 1) > train_len:
            list_data.append({"class index": 0, "folderName": folderName, "label": "Violence", "dataset":"test"})
        else:
            list_data.append({"class index": 0, "folderName": folderName, "label": "Violence", "dataset":"train"})
    keys = list_data[0].keys()
    with open(f'{dataset_root}/dataset_info.csv', 'w', newline='') as output_file:
        df = csv.DictWriter(output_file, keys)
        df.writeheader()
        df.writerows(list_data)

class Violence_Drone_Dataset(Dataset):
    def __init__(self, root_dir = DEFAULT_DATA_PATH, transform = None, train=True, useVal= False):
        self.root_dir = root_dir
        self.transform = transform
        type = "train" if train else "test"
        self.data_df = pd.read_csv(f"{self.root_dir}/dataset_info.csv").groupby("dataset").get_group(type)
        
        list_records = self.data_df.to_dict("records")
        self.data_len = len(list_records)
        
        data = []
        # if train and useVal:
            
        with tqdm(list_records, unit="folder") as listFolder:
            for idx, folderInfo in enumerate(listFolder):
                listFolder.set_description(f"Folder [{idx + 1}/{self.data_len}]")
                data_point = []
                folderName = folderInfo["folderName"]
                for frame_idx in range(NUM_IMG_PER_SEQ):
                    numpy_img = cv2.imread(f"{self.root_dir}/{folderName}/{folderName}_frame_{frame_idx + 1}.jpg")
                    data_point.append(
                        numpy_img
                    )
                data.append(data_point)
                time.sleep(0.1)
        self.data = data
        self.targets = np.array([
            folderInfo["class index"]
            for folderInfo in self.data_df.iloc
        ])
        self.classes = np.array(np.array(["Violence", "Non Violence"]))
        
        print(f"Found {self.data_len} data of type {'Train' if train else 'Test'}")
        
    def __len__(self):
        return self.data_len
    def __getitem__(self, index):    
        images = self.data[index]
        labels = self.targets[index]
        
        if self.transform:
            images = self.transform(images)
        return images, labels
    


class MergeChannelTransForm(object):
    """
    Merge sequences of 15 images from dimension (15, h, w, 3) => (h, w, 45) 
    """
    def __init__(self):
        pass
    def __call__(self, seq):
        return np.dstack(tuple(seq.copy()))
    

class MixupTransform(object):
    """
    Mixup
    """  
    def __init__(self, alpha = 0.5):
        self.alpha = alpha
    def __call__(self, seq):
        alpha = self.alpha
        l = np.random.beta(alpha, alpha, 1)
        seq_clone = seq.copy()
        middleImg = seq_clone[7]
        del seq_clone[7]
        result = middleImg * l + seq_clone[0]*(1-l)
        for img in seq_clone[0:]:
            result = result * l + (1-l) * img
        
        return result.astype(np.uint8)

# a = Violence_Drone_Dataset(train=False, transform = MergeChannelTransForm())
# a = Violence_Drone_Dataset(train=False, transform=MixupTransform())

# count = 0
# for img, label in a:
#     if count == 3:
#         break
#     count +=1
#     cv2.imshow(f"Frame{count}", img)
#     cv2.waitKey(0)

