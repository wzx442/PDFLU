# # coding: UTF-8
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from models.Model_base import MyModel
# import torchvision
# from torchvision.models import ResNet18_Weights

# # from models.ResNet import ResNet18

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential( # 创建left
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )

#         self.shortcut = nn.Sequential() # 创建shortcut
#         if stride != 1 or in_channels != out_channels: # 如果stride不为1或in_channels不等于out_channels
#             self.shortcut = nn.Sequential( # 创建shortcut
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#         # 移除dropout，ResNet通常不使用dropout

#     def forward(self, x): # 前向传播
#         out = self.left(x) # 创建out
#         out += self.shortcut(x) # 创建out
#         out = F.relu(out) # 创建out
#         return out


# class Model(MyModel): # 创建Model
#     def __init__(self, config): # 初始化
#         super(Model, self).__init__() # 初始化
#         self.num_classes = config.num_classes # 创建num_classes
        
#         # 创建适合32x32输入的ResNet18
#         self.model = torchvision.models.resnet18(weights=None) # 不使用预训练权重
        
#         # 修改第一层卷积以接受32x32输入
#         # 原始ResNet18第一层是7x7卷积，步长为2，输出64通道
#         # 对于32x32输入，我们使用3x3卷积，步长为1，输出64通道
#         self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
#         # 移除maxpool层，因为32x32输入太小
#         self.model.maxpool = nn.Identity()
        
#         # 修改全连接层
#         num_ftrs = self.model.fc.in_features # 创建num_ftrs
#         self.model.fc = nn.Linear(num_ftrs, self.num_classes) # 创建model.fc

#     def forward(self, x): # 前向传播
#         return self.model(x) # 返回model(x)

# # 创建一个Model2类，实现ResNet34
# class Model2(MyModel): # 创建Model2
#     def __init__(self, config): # 初始化
#         super(Model2, self).__init__() # 初始化
#         self.num_classes = config.num_classes # 创建num_classes
#         # 创建适合32x32输入的ResNet34
#         self.model = torchvision.models.resnet34(weights=None) # 不使用预训练权重

#         # 修改第一层卷积以接受32x32输入
#         # 原始ResNet34第一层是7x7卷积，步长为2，输出64通道
#         # 对于32x32输入，我们使用3x3卷积，步长为1，输出64通道
#         self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

#         # 移除maxpool层，因为32x32输入太小
#         self.model.maxpool = nn.Identity()

#         # 修改全连接层
#         num_ftrs = self.model.fc.in_features # 创建num_ftrs
#         self.model.fc = nn.Linear(num_ftrs, self.num_classes) # 创建model.fc

#     def forward(self, x): # 前向传播
#         return self.model(x) # 返回model(x) 


# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Model_base import MyModel
import torchvision


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential( # 创建left
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential() # 创建shortcut
        if stride != 1 or in_channels != out_channels: # 如果stride不为1或in_channels不等于out_channels
            self.shortcut = nn.Sequential( # 创建shortcut
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout(0.5) # 创建dropout

    def forward(self, x): # 前向传播
        out = self.left(x) # 创建out
        out += self.shortcut(x) # 创建out
        out = F.relu(out) # 创建out
        out = self.dropout(out) # 创建out
        return out


class Model(MyModel): # 创建Model
    def __init__(self, config): # 初始化
        super(Model, self).__init__() # 初始化
        self.num_classes = config.num_classes # 创建num_classes
        
        # 创建适合32x32输入的ResNet18
        self.model = torchvision.models.resnet18(weights=None) # 不使用预训练权重
        
        # 修改第一层卷积以接受32x32输入
        # 原始ResNet18第一层是7x7卷积，步长为2，输出64通道
        # 对于32x32输入，我们使用3x3卷积，步长为1，输出64通道
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除maxpool层，因为32x32输入太小
        self.model.maxpool = nn.Identity()
        
        # 修改全连接层
        num_ftrs = self.model.fc.in_features # 创建num_ftrs
        self.model.fc = nn.Linear(num_ftrs, self.num_classes) # 创建model.fc

    def forward(self, x): # 前向传播
        return self.model(x) # 返回model(x)
    
