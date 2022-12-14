import torch
from torchvision.datasets import MNIST
import torchvision

dataset = MNIST(root="./dataset",download=True, train=False)

count  =0
for img, label in dataset:
    count +=1
    if count == 10:
        break
    print(torchvision.transforms.ToTensor()(img))