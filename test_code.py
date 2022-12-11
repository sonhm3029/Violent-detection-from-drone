import cv2
import pandas as pd
import numpy as np
import ast
import time
import os

# img = cv2.imread("meme.jpg", cv2.IMREAD_COLOR)
# print(img.shape)
# csv_img = []

# for col_id in range(img.shape[1]):
#     csv_img.append(img[:, col_id].tolist())
    
# data_dict = {key: csv_img[key] for key in range(img.shape[1])}

# df = pd.DataFrame(data_dict)

# df.to_csv("test_code.csv", index=False)
t1 = time.time()
df = pd.read_csv("test_code.csv")
df = df.to_numpy()   


new_df = []
for row_idx in range(df.shape[0]):
    new_df.append([])
    for col_idx in range(df.shape[1]):
        new_df[row_idx].append(ast.literal_eval(df[row_idx, col_idx]))
t2 = time.time()
print(f"Time to read in .csv file of one image matrix {t2-t1}")



t1 = time.time()

DIR = "images"
arr = []

for file in os.listdir(DIR):
    img = cv2.imread(f"{DIR}/{file}")
    arr.append(img)

print(len(arr))
t2 = time.time()
print(f"Time to read in 15 images file directly by opencv {t2-t1}")

