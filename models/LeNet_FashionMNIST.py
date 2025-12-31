# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Model_base import MyModel

class Model(MyModel): # 创建Model
    def __init__(self, config): # 初始化
        super(Model, self).__init__() # 初始化
        self.conv1 = nn.Conv2d(1, 6, 5) # 创建conv1
        self.relu = nn.ReLU() # 创建relu
        self.maxpool1 = nn.MaxPool2d(2, 2) # 创建maxpool1
        self.conv2 = nn.Conv2d(6, 16, 5) # 创建conv2
        self.maxpool2 = nn.MaxPool2d(2, 2) # 创建maxpool2
        self.fc1 = nn.Linear(256, 120) # 创建fc1
        self.fc2 = nn.Linear(120, 84) # 创建fc2
        self.fc3 = nn.Linear(84, config.num_classes) # 创建fc3

    def forward(self, x): # 前向传播
        x = self.conv1(x) # 创建x
        x = self.relu(x) # 创建x
        x = self.maxpool1(x) # 创建x
        x = self.conv2(x) # 创建x
        x = self.maxpool2(x) # 创建x
        x = x.view(x.size(0), -1) # 创建x
        x = F.relu(self.fc1(x)) # 创建x
        mid_val = F.relu(self.fc2(x)) # 创建mid_val
        output = self.fc3(mid_val) # 创建output
        return output # 返回output