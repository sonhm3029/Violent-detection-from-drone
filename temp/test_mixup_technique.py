import numpy as np
import cv2
import os

# folder = "data/2022_12_12_16707813209779"

# seq = []

# for fileName in os.listdir(folder):
#     seq.append(cv2.imread(f"{folder}/{fileName}"))
    

# print(seq)
alpha = 0.5

l = np(alpha, alpha)

img1 = cv2.imread("temp/meme.jpg")
img2 = cv2.imread("temp/stonks.jpg")


img1 = cv2.resize(img1, (640, 640), interpolation=cv2.INTER_LINEAR)
img2 = cv2.resize(img2, (640, 640), interpolation=cv2.INTER_LINEAR)

img = (img1* l + (1-l) * img2).astype("uint8")

cv2.imshow("frame",img)

cv2.waitKey(0)