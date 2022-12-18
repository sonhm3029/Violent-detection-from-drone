from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
import cv2
import numpy as np

a = np.array([
[[[ 83,  83,  83] *15,
    [ 45,  46,  60] *15],

    [[ 70,  77,  74] *15,
    [167, 180, 221] *15],

    [[192, 192, 204] *15,
    [220, 195, 159] *15]],
[[[ 183,  183,  83] *15,
    [ 45,  146,  60] *15],

    [[ 270,  177,  74] *15,
    [167, 180, 221] *15],

    [[192, 192, 204] *15,
    [220, 195, 159] *15]]
])

print(a)
print((a/255.0))
print("MEAN")
print((a/255.0).mean(axis=(0,1,2)))

# img = cv2.imread("temp/meme.jpg")

# print(cv2.resize(img, (2, 3)))

# img = transforms.ToTensor()(img)
# img = transforms.Resize((2, 3),interpolation=InterpolationMode.BICUBIC)(img)

# print(img)

# img = transforms.Normalize((0.1, 0.2, 0.3), (0.1, 0.1, 0.1))(img)

# print(img)