# coding: UTF-8
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import huggingface_hub
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.Model_base import MyModel
import torchvision
import timm
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import ViTForImageClassification



class ViT_Cifar100(nn.Module): # 创建ViT_Cifar100
    def __init__(self, config): # 初始化
        super(ViT_Cifar100, self).__init__() # 初始化
        self.num_classes = config.num_classes # 创建num_classes
        self.model = timm.create_model('vit_small_patch16_224', pretrained=True) # 创建model
        self.model.head = nn.Linear(self.model.head.in_features, self.num_classes) # 创建model.head

    def forward(self, x): # 前向传播
        return self.model(x) # 返回model(x)
