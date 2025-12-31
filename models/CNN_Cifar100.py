# # coding: UTF-8
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from models.Model_base import MyModel
# import torchvision
# import timm
# from torchvision.models import ResNet18_Weights

# class ResidualBlock(nn.Module): # 创建ResidualBlock
#     def __init__(self, in_channels, out_channels, stride=1): # 初始化
#         super(ResidualBlock, self).__init__() # 初始化
#         self.left = nn.Sequential( # 创建left
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), # 创建Conv2d
#             nn.BatchNorm2d(out_channels), # 创建BatchNorm2d
#             nn.ReLU(inplace=True), # 创建ReLU
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # 创建Conv2d
#             nn.BatchNorm2d(out_channels) # 创建BatchNorm2d
#         )

#         self.shortcut = nn.Sequential() # 创建shortcut
#         if stride != 1 or in_channels != out_channels: # 如果stride不为1或in_channels不等于out_channels
#             self.shortcut = nn.Sequential( # 创建shortcut
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), # 创建Conv2d
#                 nn.BatchNorm2d(out_channels) # 创建BatchNorm2d
#             )
#         self.dropout = nn.Dropout(0.5) # 创建dropout

#     def forward(self, x): # 前向传播
#         out = self.left(x) # 创建out
#         out += self.shortcut(x) # 创建out
#         out = F.relu(out) # 创建out
#         out = self.dropout(out) # 创建out
#         return out



# class Model(MyModel): # 创建Model
#     def __init__(self, config): # 初始化
#         super(Model, self).__init__() # 初始化
#         self.num_classes = config.num_classes # 创建num_classes
        
#         # 创建适合32x32输入的ResNet18
#         self.model = torchvision.models.resnet18(weights=None) # 不使用预训练权重
#         # self.model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # 不使用预训练权重

        
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



# # 创建一个Model2类，实现ResNet34，修改网络结构，以适应cifar100数据集
# class Model2(MyModel): # 创建Model2
#     def __init__(self, config): # 初始化
#         super(Model2, self).__init__() # 初始化
#         self.num_classes = config.num_classes # 创建num_classes
#         self.model = torchvision.models.resnet34(weights=None) # 不使用预训练权重
#         # self.model = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) # 不使用预训练权重

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
import timm

class ResidualBlock(nn.Module): # 创建ResidualBlock
    def __init__(self, in_channels, out_channels, stride=1): # 初始化
        super(ResidualBlock, self).__init__() # 初始化
        self.left = nn.Sequential( # 创建left
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), # 创建Conv2d
            nn.BatchNorm2d(out_channels), # 创建BatchNorm2d
            nn.ReLU(inplace=True), # 创建ReLU
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # 创建Conv2d
            nn.BatchNorm2d(out_channels) # 创建BatchNorm2d
        )

        self.shortcut = nn.Sequential() # 创建shortcut
        if stride != 1 or in_channels != out_channels: # 如果stride不为1或in_channels不等于out_channels
            self.shortcut = nn.Sequential( # 创建shortcut
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), # 创建Conv2d
                nn.BatchNorm2d(out_channels) # 创建BatchNorm2d
            )
        self.dropout = nn.Dropout(0.5) # 创建dropout

    def forward(self, x): # 前向传播
        out = self.left(x) # 创建out
        out += self.shortcut(x) # 创建out
        out = F.relu(out) # 创建out
        out = self.dropout(out) # 创建out
        return out


# class Model(MyModel):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.layer1 = self.make_layer(ResidualBlock, 64, 3, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)
#         self.fc = nn.Linear(512, config.num_classes)
#
#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         mid_val = out.view(out.size(0), -1)
#         out = self.fc(mid_val)
#         return out, mid_val

class Model(MyModel): # 创建Model
    def __init__(self, config): # 初始化
        super(Model, self).__init__() # 初始化
        self.num_classes = config.num_classes # 创建num_classes
        self.model = torchvision.models.resnet18(weights=None) # 创建model
        num_ftrs = self.model.fc.in_features # 创建num_ftrs
        self.model.fc = nn.Linear(num_ftrs, self.num_classes) # 创建model.fc


    def forward(self, x): # 前向传播
        return self.model(x) # 返回model(x)

# class Model(MyModel):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         self.num_classes = config.num_classes
#         self.model = timm.create_model('vit_small_patch16_224', pretrained=True)
#         # self.model = ViTForImageClassification.from_pretrained('google/vit-small-patch16-224-in21k', num_labels=self.num_classes)
        
#         # processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
#         # self.model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
#         self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)

#     def forward(self, x):
#         return self.model(x)
