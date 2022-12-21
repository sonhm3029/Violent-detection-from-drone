from utils.datasets import splitDataset
import argparse


def split():
    train_ratio = opt.train_ratio
    dataset_root = opt.data_source
    
    splitDataset(train_ratio=train_ratio, dataset_root=dataset_root)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-source',type=str, default='../../data')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training dataset ratio')
    opt = parser.parse_args()
    print(opt) 
    
    split()