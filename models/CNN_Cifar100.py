import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Model_base import MyModel
import torchvision
import timm

class ResidualBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, stride=1): 
        super(ResidualBlock, self).__init__() 
        self.left = nn.Sequential( 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels) 
        )

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_channels != out_channels: 
            self.shortcut = nn.Sequential( 
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(out_channels) 
            )
        self.dropout = nn.Dropout(0.5) 

    def forward(self, x): 
        out = self.left(x) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        out = self.dropout(out) 
        return out



class Model(MyModel): 
    def __init__(self, config): 
        super(Model, self).__init__() 
        self.num_classes = config.num_classes 
        self.model = torchvision.models.resnet18(weights=None) 
        num_ftrs = self.model.fc.in_features 
        self.model.fc = nn.Linear(num_ftrs, self.num_classes) 


    def forward(self, x): 
        return self.model(x) 
