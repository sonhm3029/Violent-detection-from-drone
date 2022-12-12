import os
import pandas as pd
from skimage import io
import time
import cv2


ROOT_PATH = "data"

# Get all data at start
# t1 = time.time()
# data = []
# for folderName in os.listdir(ROOT_PATH):
#     data_point = []
#     for fileName in os.listdir(f"{ROOT_PATH}/{folderName}"):
#         data_point.append(cv2.imread(f"{ROOT_PATH}/{folderName}/{fileName}"))
#     data.append(data_point)
    
# for dtp in data:
#     print(len(dtp))
# t2 = time.time()

# print(t2-t1)
# Get data by data when training
t1 = time.time()
data = pd.DataFrame(os.listdir(ROOT_PATH))
data.to_csv(f"{ROOT_PATH}/dataset.csv", index=False)

df = pd.read_csv(f"{ROOT_PATH}/dataset.csv")

dataset = list(df.loc[:,'0'].to_numpy()[:-1])


t2 = time.time()
print(t2-t1)
