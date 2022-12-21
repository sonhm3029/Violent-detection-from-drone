import argparse
import time
import matplotlib.pyplot as plt

from models.efficientnet import *
from utils.eff_datasets import *


import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
from torchsummary import summary

def plotting(colors: tuple, labels : tuple, savefigName: str, data: tuple, figsize=(10, 7)):
    train, val = data
    color_train, color_val = colors
    label_train, label_val = labels
    
    plt.figure(figsize=figsize)
    plt.plot(train, color=color_train, label=label_train)
    plt.plot(val, color=color_val, label=label_val)
    plt.legend()
    plt.savefig(savefigName)

def fit(model, dataloader, 
        epoch, epochs, device, 
        criterion, optimizer,  train=True):
    if train:
        model.train()
    else:
        model.eval() 
    
    
    running_loss = 0.0
    running_correct = 0
    n_samples = 0
    
    print("Train" if train else "Val")
    
    with tqdm(dataloader, unit='batch') as tepoch:
        for images, labels in tepoch:    
            
            tepoch.set_description(f"Epoch [{epoch}/{epochs}]")
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct = (preds == labels).sum().item()

            running_correct += (preds == labels).sum().item()

            n_samples += labels.size(0)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            tepoch.set_postfix(loss=loss.item()/labels.size(0), accuracy=100.*correct/labels.size(0))
            
        process_loss = running_loss / n_samples
        process_acc = 100. * running_correct / n_samples
    
    return process_loss, process_acc
                

def train():
    # Constant
    ROOT_DIR = opt.data_source
    device = torch.device(opt.device)
    IMG_SIZE = (opt.img_size, opt.img_size)
    save_model_epoch = opt.save_model_epoch
    
    # Hyper parameter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.lr_rate
    
    # Model loaded in
    model = ViolenceEfficientNet()
    summary(model, (45, 224, 224))
    
    MEAN, STD = [0.51551778, 0.43288471, 0.44265668]*15, [0.19281362, 0.1960019 , 0.20439348]*15
    
    data_transforms = transforms.Compose([
        MergeChannelTransForm(),
        transforms.ToTensor(),
        transforms.Resize(IMG_SIZE,interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize(MEAN, STD)
    ])
    
    train_dataset = Violence_Drone_Dataset(root_dir=ROOT_DIR,
                                           train=True, transform=data_transforms)
    test_dataset = Violence_Drone_Dataset(root_dir=ROOT_DIR,
                                          train=False, transform=data_transforms) 
    
    train_loader = DataLoader(dataset=train_dataset,
                              shuffle=True, batch_size=batch_size,
                              drop_last=True)
    test_loader = DataLoader(dataset= test_dataset,
                             shuffle=False, batch_size=batch_size,
                             drop_last=False)  
    
    # Preparing for training
    optimizer = optimizer = torch.optim.AdamW(model.parameters(),
                              lr=learning_rate,
                              weight_decay=0.0005)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 40, 60, 80, 100],
        gamma=0.2
    )
    criterion = nn.CrossEntropyLoss()
    
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    
    
    # Training and validation
    model = model.to(device)
    
    start = time.time()
    
    for epoch in range(epochs):
        train_epoch_loss, train_epoch_acc = fit(model=model, dataloader=train_loader,
                                                epoch=epoch, epochs=epochs,
                                                device=device, criterion=criterion,
                                                optimizer=optimizer, train=True)
        exp_lr_scheduler.step()
    
        val_epoch_loss, val_epoch_acc = fit(model=model, dataloader=test_loader,
                                                epoch=epoch, epochs=epochs,
                                                device=device, criterion=criterion,
                                                optimizer=optimizer, train=False)
    
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_acc)
    
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_acc)
        
        if( (epoch + 1) % save_model_epoch == 0):
            new_run_idx = 1
            if not os.path.exists("eff_runs"):
                os.mkdir("eff_runs")
            else:
                num_runs_folder = len(os.listdir("eff_runs"))
                new_run_idx = num_runs_folder + 1
            
            runs_folder_name = f"eff_runs/exp_{new_run_idx}"
            os.mkdir(runs_folder_name)
                
            torch.save(model.state_dict, f"{runs_folder_name}/last_weights.pth")
            
            # Plotting loss
            plotting(colors=('orange', 'red'),
                    labels=("train loss", "val loss"),
                    savefigName=f"{runs_folder_name}/loss.png",
                    data=(train_loss, val_loss))
            # Ploting acc
            plotting(colors=('green', 'blue'),
                    labels=("train acc", "val acc"),
                    savefigName=f"{runs_folder_name}/acc.png",
                    data=(train_accuracy, val_accuracy))
            
            print("Model saved!")
    
    end = time.time()
    print((end-start)/60, 'minutes')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--data-source',type=str, default='../../data')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size in dataloader')
    parser.add_argument('--lr-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save-model-epoch', type=int, default=100, help='Save model per epochs')
    opt = parser.parse_args()
    print(opt)  
    
    train()