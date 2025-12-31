#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
import torchvision
from peft import LoraConfig, get_peft_model
import timm

# from utils import *
class MyModel(nn.Module): # 创建MyModel
    def __init__(self): # 初始化
        super(MyModel, self).__init__() # 初始化

    @staticmethod
    def split_weight_name(name): # 分割权重名称
        if 'weight' or 'bias' in name: # 如果weight或bias在name中
            return ''.join(name.split('.')[:-1]) # 返回name的分割结果
        return name # 返回name

    def save_params(self): # 保存参数
        for param_name, param in self.named_parameters(): # 遍历参数
            if 'alpha' in param_name or 'beta' in param_name: # 如果alpha或beta在param_name中
                continue # 继续
            _buff_param_name = param_name.replace('.', '__') # 创建_buff_param_name
            self.register_buffer(_buff_param_name, param.data.clone()) # 注册缓冲区

    def compute_diff(self): # 计算差异
        diff_mean = dict() # 创建diff_mean
        for param_name, param in self.named_parameters(): # 遍历参数
            layer_name = self.split_weight_name(param_name) # 创建layer_name
            _buff_param_name = param_name.replace('.', '__') # 创建_buff_param_name
            old_param = getattr(self, _buff_param_name, default=0.0) # 创建old_param
            diff = (param - old_param) ** 2 # 创建diff
            diff = diff.sum() # 创建diff
            total_num = reduce(lambda x, y: x*y, param.shape) # 创建total_num
            diff /= total_num # 创建diff
            diff_mean[layer_name] = diff # 创建diff_mean
        return diff_mean # 返回diff_mean

    def remove_grad(self, name=''): # 删除梯度
        for param_name, param in self.named_parameters(): # 遍历参数
            if name in param_name: # 如果name在param_name中
                param.requires_grad = False # 设置param.requires_grad为False


class Lora(nn.Module): # 创建Lora
    def __init__(self, args, global_model, alpha): # 初始化
        super(Lora, self).__init__() # 初始化
        if args.data_name == 'cifar10': # 如果args.data_name为cifar10
            if args.resnet == 18:
                global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) # 加载global_model
                target_modules = ["layer3.0.conv1", "layer3.0.conv2", "layer3.1.conv1", "layer3.1.conv2", "layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"] # 创建target_modules
            elif args.resnet == 34:
                global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) # 加载global_model
                target_modules = ["layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv2", "layer2.0.conv2",  "layer2.1.conv1", "layer2.1.conv2", "layer2.2.conv1", "layer3.0.conv1", "layer3.0.conv2","layer4.0.conv2", "layer4.1.conv1","layer4.1.conv2","layer4.2.conv1","layer4.2.conv2","fc"] # 创建target_modules

#             target_modules = [
#     "to_patch_embedding.1",  # nn.Linear in to_patch_embedding
#     "transformer.layers.0.0.to_qkv",  # Attention layer's to_qkv in the first transformer block
#     "transformer.layers.0.0.to_out",  # Attention layer's to_out in the first transformer block
#     "transformer.layers.0.1.net.1",  # First Linear layer in FeedForward in the first transformer block
#     "transformer.layers.0.1.net.3",  # Second Linear layer in FeedForward in the first transformer block
#     "transformer.layers.1.0.to_qkv",  # Attention layer's to_qkv in the second transformer block
#     "transformer.layers.1.0.to_out",  # Attention layer's to_out in the second transformer block
#     "transformer.layers.1.1.net.1",  # First Linear layer in FeedForward in the second transformer block
#     "transformer.layers.1.1.net.3",  # Second Linear layer in FeedForward in the second transformer block
#     # Add more layers as needed for deeper transformer blocks
#     "linear_head.1"  # nn.Linear in linear_head
# ]

        elif args.data_name == 'cifar100': # 如果args.data_name为cifar100
            if args.resnet == 18:
                global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) # 加载global_model
                target_modules = ["layer2.0.conv1","layer3.4.conv1", "layer3.4.conv2", "layer3.5.conv1", "layer3.5.conv2", "layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "layer4.2.conv1","layer4.2.conv2","fc"] # 创建target_modules

        elif args.data_name == 'fashionmnist': # 如果args.data_name为fashionmnist
            global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) # 加载global_model

            target_modules = ['conv1', 'fc3'] # 创建target_modules
        # elif args.data_name == 'adult': # 如果args.data_name为adult
        #     global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) # 加载global_model
        #     target_modules = ['fc3'] # 创建target_modules
        # elif args.data_name == 'text': # 如果args.data_name为text
        #     global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) # 加载global_model
        #     target_modules = ["encoder.layer.11.output.dense"] # 创建target_modules



        # M_j = (alpha*M^u +(1-alpha)*M^r) \odot M^{rst}
        # alpha / (1-alpha) = lora_alpha / r
        # r*(alpha / (1-alpha)) = lora_alpha
        r=16
        lora_alpha = int(r*(alpha / (1-alpha)))
        print(f"r: {r}, lora_alpha: {lora_alpha}")
        config = LoraConfig(
                    r = r, # 创建r
                    lora_alpha = lora_alpha, # 创建lora_alpha
                    target_modules = target_modules, # 创建target_modules
                    lora_dropout = 0.1, # 创建lora_dropout
                    bias = "none", # 创建bias
        )
        self.lora_model = get_peft_model(global_model, config) # 创建lora_model
        for name, param in self.lora_model.named_parameters(): # 遍历lora_model的参数
            if not any(target in name for target in config.target_modules): # 如果target不在name中
                param.requires_grad = False # 设置param.requires_grad为False, 不进行梯度更新

    def forward(self, x): # 前向传播
        return self.lora_model(x) # 返回lora_model(x)