import torch
import sys

def calculate_model_size(model):
    model_size = 0
    for param in model.parameters():
        model_size += param.numel()
    model_size = model_size * 4 / 1024 / 1024
    return model_size


def calculate_model_size_specific(model, args):
    model_size = 0
    target_modules = []
    
    if args.data_name == 'cifar10': 
        if args.resnet == 18:
            target_modules = ["layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"] 
        elif args.resnet == 34:
            target_modules = ["layer4.1.conv2","layer4.2.conv1","layer4.2.conv2","fc"] 

    elif args.data_name == 'cifar100': 
        if args.resnet == 18:
                model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) 
                target_modules = ["layer4.1.conv1", "layer4.1.conv2", "layer4.2.conv1","layer4.2.conv2","fc"] 

    elif args.data_name == 'fashionmnist': 
        target_modules = ['conv1', 'fc3'] 
    
    # print(f"Target modules: {target_modules}")
    # print("All model parameters:")
    matched_params = []
    
    for name, param in model.named_parameters():
        # print(f"  {name}: {param.numel()} parameters")
        for target in target_modules:
            if target in name:  
                # print(f"  ✓ Matched: {name} contains {target}")
                model_size += param.numel()
                matched_params.append(name)
                break
    
    # print(f"Matched parameters: {matched_params}")
    # print(f"Total matched parameter count: {sum(param.numel() for name, param in model.named_parameters() if any(target in name for target in target_modules))}")
    
    model_size = model_size * 4 / 1024 / 1024
    return model_size
