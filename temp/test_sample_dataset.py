import torch
from torchvision.datasets import MNIST

dataset = MNIST(root="./dataset",download=True, train=False)

count  =0
for img in dataset:
    count +=1
    if count == 10:
        break
    print(img)