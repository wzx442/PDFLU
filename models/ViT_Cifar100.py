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



class ViT_Cifar100(nn.Module): 
    def __init__(self, config): 
        super(ViT_Cifar100, self).__init__() 
        self.num_classes = config.num_classes 
        self.model = timm.create_model('vit_small_patch16_224', pretrained=True) 
        self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)

    def forward(self, x): 
        return self.model(x) 