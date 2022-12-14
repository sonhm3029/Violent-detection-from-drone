import torch
import torchvision
import torch.nn as nn
from torchvision.models import efficientnet_b0

def freezeModel(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

class ViolenceEfficientNet(nn.Module):
    def __init__(self):
        super(ViolenceEfficientNet, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=45,
            out_channels=3,
            stride=1,
            padding=1,
            kernel_size=3
        )
        self.effNet = freezeModel(efficientnet_b0(pretrained=True))
        self.effNet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=640, bias=True),
            nn.Linear(in_features=640, out_features=3, bias=True)
        )
        
    def forward(self, x):
        output = self.conv(x)
        output = self.effNet(output)

        return output    
    
model = efficientnet_b0(pretrained=True)
print(model.classifier)