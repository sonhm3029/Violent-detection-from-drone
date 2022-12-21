import argparse
import time


from .model import *
from .utils.datasets import *


import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode



def train():
    # Constant
    ROOT_DIR = opt.data_source
    device = torch.device(opt.device)
    IMG_SIZE = (opt.img_size, opt.img_size)
    
    # Hyper parameter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.lr_rate
    
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--data-source',type=str, default='../../data')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size in dataloader')
    parser.add_argument('--lr-rate', type=float, default=0.0001, help='Learning rate')
    opt = parser.parse_args()
    print(opt)  
     