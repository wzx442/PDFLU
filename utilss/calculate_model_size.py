# 输入是global_model，输出是模型大小，单位是MB
import torch
import sys

def calculate_model_size(model):
    model_size = 0
    for param in model.parameters():
        model_size += param.numel()
    model_size = model_size * 4 / 1024 / 1024
    return model_size


# 创建一个模型计算函数，只计算特定名称的层所占据的大小，单位是MB
def calculate_model_size_specific(model, args):
    model_size = 0
    target_modules = []
    
    if args.data_name == 'cifar10': # 如果args.data_name为cifar10
        if args.resnet == 18:
            target_modules = ["layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"] # 创建target_modules
        elif args.resnet == 34:
            target_modules = ["layer4.1.conv2","layer4.2.conv1","layer4.2.conv2","fc"] # 创建target_modules

    elif args.data_name == 'cifar100': # 如果args.data_name为cifar100
        if args.resnet == 18:
                model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(args.data_name, args.file_name))) # 加载global_model
                target_modules = ["layer4.1.conv1", "layer4.1.conv2", "layer4.2.conv1","layer4.2.conv2","fc"] # 创建target_modules

    elif args.data_name == 'fashionmnist': # 如果args.data_name为fashionmnist
        target_modules = ['conv1', 'fc3'] # 创建target_modules
    
    # 调试：打印所有参数名称和目标模块
    # print(f"Target modules: {target_modules}")
    # print("All model parameters:")
    matched_params = []
    
    for name, param in model.named_parameters():
        # print(f"  {name}: {param.numel()} parameters")
        # 检查完整匹配和部分匹配
        for target in target_modules:
            if target in name:  # 使用 in 而不是 == 来匹配
                # print(f"  ✓ Matched: {name} contains {target}")
                model_size += param.numel()
                matched_params.append(name)
                break
    
    # print(f"Matched parameters: {matched_params}")
    # print(f"Total matched parameter count: {sum(param.numel() for name, param in model.named_parameters() if any(target in name for target in target_modules))}")
    
    model_size = model_size * 4 / 1024 / 1024
    return model_size
