import torch
import cv2
import os

FOLDER = "data/2022_12_12_16707813209779"

seq = []

for fileName in os.listdir(FOLDER):
    seq.append(torch.from_numpy(cv2.imread(f"{FOLDER}/{fileName}")))
    

print(seq[0].shape)
newImg = torch.cat(tuple(seq), 2)
print(newImg)
