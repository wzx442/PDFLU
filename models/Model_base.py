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
class MyModel(nn.Module): 
    def __init__(self): 
        super(MyModel, self).__init__() 

    @staticmethod
    def split_weight_name(name): 
        if 'weight' or 'bias' in name: 
            return ''.join(name.split('.')[:-1]) 
        return name

    def save_params(self): 
        for param_name, param in self.named_parameters(): 
            if 'alpha' in param_name or 'beta' in param_name: 
                continue 
            _buff_param_name = param_name.replace('.', '__') 
            self.register_buffer(_buff_param_name, param.data.clone()) 

    def compute_diff(self): 
        diff_mean = dict() 
        for param_name, param in self.named_parameters(): 
            layer_name = self.split_weight_name(param_name) 
            _buff_param_name = param_name.replace('.', '__') 
            old_param = getattr(self, _buff_param_name, default=0.0) 
            diff = (param - old_param) ** 2 
            diff = diff.sum() 
            total_num = reduce(lambda x, y: x*y, param.shape) 
            diff /= total_num 
            diff_mean[layer_name] = diff 
        return diff_mean 

    def remove_grad(self, name=''): 
        for param_name, param in self.named_parameters(): 
            if name in param_name: 
                param.requires_grad = False 

class Lora(nn.Module):
    def __init__(self, args, global_model, alpha):
        super(Lora, self).__init__() 
        if args.data_name == 'cifar10':
            if args.resnet == 18:
                global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) 
                target_modules = ["layer3.0.conv1", "layer3.0.conv2", "layer3.1.conv1", "layer3.1.conv2", "layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"] 
            elif args.resnet == 34:
                global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) 
                target_modules = ["layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv2", "layer2.0.conv2",  "layer2.1.conv1", "layer2.1.conv2", "layer2.2.conv1", "layer3.0.conv1", "layer3.0.conv2","layer4.0.conv2", "layer4.1.conv1","layer4.1.conv2","layer4.2.conv1","layer4.2.conv2","fc"] 
        elif args.data_name == 'cifar100': 
            if args.resnet == 18:
                global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) 
                target_modules = ["layer2.0.conv1","layer3.4.conv1", "layer3.4.conv2", "layer3.5.conv1", "layer3.5.conv2", "layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "layer4.2.conv1","layer4.2.conv2","fc"] 

        elif args.data_name == 'fashionmnist': 
            global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) 

            target_modules = ['conv1', 'fc3'] 
        r=16
        lora_alpha = int(r*(alpha / (1-alpha)))
        print(f"r: {r}, lora_alpha: {lora_alpha}")
        config = LoraConfig(
                    r = r, 
                    lora_alpha = lora_alpha, 
                    target_modules = target_modules, 
                    lora_dropout = 0.1, 
                    bias = "none", 
        )
        self.lora_model = get_peft_model(global_model, config) 
        for name, param in self.lora_model.named_parameters(): 
            if not any(target in name for target in config.target_modules): 
                param.requires_grad = False 

    def forward(self, x): 
        return self.lora_model(x) 