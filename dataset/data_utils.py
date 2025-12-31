import random

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
import copy
from models.cutout import Cutout

train_size = 0.99 # merge original training set and test set, then split it manually.
least_samples = 100 # guarantee that each client must have at least one samples for testing.



def data_set(data_name):
    if (data_name == 'mnist'):
        trainset = datasets.MNIST('./dataset/mnist', train=True, download=True, # 下载MNIST数据集
                                  transform=transforms.Compose([
                                      transforms.ToTensor(), # 转换为张量
                                      transforms.Normalize((0.1307,), (0.3081,)) # 归一化
                                  ]))

        testset = datasets.MNIST('./dataset/mnist', train=False, download=True, # 下载MNIST数据集
                                 transform=transforms.Compose([
                                     transforms.ToTensor(), # 转换为张量
                                     transforms.Normalize((0.1307,), (0.3081,)) # 归一化
                                 ]))
    elif (data_name == 'fashionmnist'):
        trainset = datasets.MNIST('./dataset/fashionmnist', train=True, download=True, # 下载FashionMNIST数据集
                                  transform=transforms.Compose([
                                      transforms.ToTensor(), # 转换为张量
                                      transforms.Normalize((0.1307,), (0.3081,)) # 归一化
                                  ]))

        testset = datasets.MNIST('./dataset/fashionmnist', train=False, download=True, # 下载FashionMNIST数据集
                                 transform=transforms.Compose([
                                     transforms.ToTensor(), # 转换为张量    
                                     transforms.Normalize((0.1307,), (0.3081,)) # 归一化
                                 ]))
    # model: ResNet-18
    elif (data_name == 'cifar10'):
        # transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4), # 随机裁剪
        #     transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
        #     transforms.ToTensor(), # 转换为张量
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 归一化
        # ])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # 测试集不使用数据增强
        test_transform = transforms.Compose([
            transforms.ToTensor(), # 转换为张量
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 归一化
        ])
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        #     transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #R,G,B每层的归一化用到的均值和方差
        #     Cutout(n_holes=1, length=16),
        # ])

        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        # ])

        trainset = datasets.CIFAR10(root='./dataset/cifar10', train=True, download=True, transform=train_transform)# 下载CIFAR-10数据集

        testset = datasets.CIFAR10(root='./dataset/cifar10', train=False, download=True, transform=test_transform)# 下载CIFAR-10数据集

    elif (data_name == 'cifar100'):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), # 随机裁剪
            transforms.RandomHorizontalFlip(), # 随机水平翻转
            transforms.ToTensor(), # 转换为张量
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 归一化
        ])

        trainset = datasets.CIFAR100('./dataset/cifar100', train=True, download=True, transform=transform)# 下载CIFAR-100数据集

        testset = datasets.CIFAR100('./dataset/cifar100', train=False, download=True, transform=transform)# 下载CIFAR-100数据集

    return trainset, testset

def separate_data(data, num_clients, num_classes, args, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)] # 创建X
    y = [[] for _ in range(num_clients)] # 创建y

    statistic = [[] for _ in range(num_clients)] # 创建statistic

    dataset_content, dataset_label = data # 创建dataset_content和dataset_label

    dataidx_map = {} # 创建dataidx_map

    classes_ls = [i for i in range(num_classes)] # 创建classes_ls

    if not niid: # 如果niid为False
        partition = 'pat' # 创建partition
        class_per_client = len(classes_ls) # 创建class_per_client

    if partition == 'pat': # 如果partition为pat
        idxs = np.array(range(len(dataset_label))) # 创建idxs
        idx_for_each_class = [] # 创建idx_for_each_class
        for cls in classes_ls: # 遍历classes_ls 
            idx_for_each_class.append(idxs[dataset_label == cls]) # 创建idx_for_each_class

        class_num_per_client = [class_per_client for _ in range(num_clients)] # 创建class_num_per_client
        for i in classes_ls: # 遍历classes_ls
            selected_clients = [] # 创建selected_clients
            for client in range(num_clients): # 遍历num_clients
                if class_num_per_client[client] > 0: # 如果class_num_per_client大于0    
                    selected_clients.append(client) # 添加client
            selected_clients = selected_clients[:int(np.ceil((num_clients /len(classes_ls)) *class_per_client))] # 创建selected_clients

            num_all_samples = len(idx_for_each_class[i]) # 创建num_all_samples
            num_selected_clients = len(selected_clients) # 创建num_selected_clients
            num_per = num_all_samples / num_selected_clients # 创建num_per
            if balance: # 如果balance为True
                num_samples = [int(num_per) for _ in range(num_selected_clients -1)] # 创建num_samples
            else:
                num_samples = np.random.randint(max(num_per /10, least_samples /len(classes_ls)), num_per, num_selected_clients -1).tolist() # 创建num_samples
            num_samples.append(num_all_samples -sum(num_samples)) # 创建num_samples

            idx = 0 # 创建idx
            for client, num_sample in zip(selected_clients, num_samples): 
                if client not in dataidx_map.keys(): # 如果client不在dataidx_map中
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample] # 创建dataidx_map
                else: # 如果client在dataidx_map中
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx +num_sample], axis=0) # 创建dataidx_map
                idx += num_sample
                class_num_per_client[client] -= 1 # 创建class_num_per_client

    elif partition == "dir": # 如果partition为dir
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0 # 创建min_size
        K = len(classes_ls) # 创建K
        N = len(dataset_label) # 创建N

        try_cnt = 1 # 创建try_cnt
        while min_size < least_samples: # 如果min_size小于least_samples
            if try_cnt > 1: # 如果try_cnt大于1
                print \
                    (f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.') # 打印客户端数据大小不满足最小要求

            idx_batch = [[] for _ in range(num_clients)] # 创建idx_batch
            for k in range(K): 
                idx_k = np.where(dataset_label == k)[0] # 创建idx_k
                np.random.shuffle(idx_k) # 随机打乱idx_k
                proportions = np.random.dirichlet(np.repeat(args.alpha, num_clients)) # 创建proportions
                proportions = np.array([ p *(len(idx_j ) < N /num_clients) for p ,idx_j in zip(proportions ,idx_batch)]) # 创建proportions
                proportions = proportions /proportions.sum() # 创建proportions
                proportions = (np.cumsum(proportions ) *len(idx_k)).astype(int)[:-1] # 创建proportions
                idx_batch = [idx_j + idx.tolist() for idx_j ,idx in zip(idx_batch ,np.split(idx_k ,proportions))] # 创建idx_batch
                min_size = min([len(idx_j) for idx_j in idx_batch]) # 创建min_size
            try_cnt += 1 # 创建try_cnt

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j] # 创建dataidx_map
    else:
        raise NotImplementedError # 抛出未实现错误

    # assign data
    for client in range(num_clients): # 遍历num_clients
        idxs = dataidx_map[client] # 创建idxs
        X[client] = dataset_content[idxs] # 创建X
        y[client] = dataset_label[idxs] # 创建y

        for i in np.unique(y[client]): # 遍历np.unique(y[client])
            statistic[client].append((int(i), int(sum(y[client ]==i)))) # 创建statistic

    del data # 删除data

    for client in range(num_clients): # 遍历num_clients
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client])) # 打印客户端数据大小
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]]) # 打印客户端数据标签
        print("-" * 50) # 打印分割线

    return X, y, statistic # 返回X, y, statistic

def split_test_proxy(test_loader, args): # 分割测试代理
    test_data_x, test_data_y, proxy_data_x, proxy_data_y = [], [], [], [] # 创建test_data_x, test_data_y, proxy_data_x, proxy_data_y
    for test_data in test_loader: # 遍历test_loader
        data, label = test_data # 创建data, label
    dataset_image = [] # 创建dataset_image
    dataset_label = [] # 创建dataset_label

    dataset_image.extend(data.cpu().detach().numpy()) # 创建dataset_image
    dataset_label.extend(label.cpu().detach().numpy()) # 创建dataset_label
    dataset_image = np.array(dataset_image) # 创建dataset_image
    dataset_label = np.array(dataset_label) # 创建dataset_label
    num_classes = args.num_classes # 创建num_classes
    idxs = np.array(range(len(dataset_label))) # 创建idxs
    idx_for_each_class = [] # 创建idx_for_each_class
    for i in range(num_classes): # 遍历num_classes
        idx_for_each_class.append(idxs[dataset_label == i]) # 创建idx_for_each_class
        num_class_proxy = len(idx_for_each_class[i])*args.proxy_frac # 创建num_class_proxy
        idx_class_proxy = np.random.choice(idx_for_each_class[i], int(num_class_proxy)) # 创建idx_class_proxy
        idx_class_test = list(set(idx_for_each_class[i])-set(idx_class_proxy)) # 创建idx_class_test
        proxy_data_x.extend(dataset_image[idx_class_proxy]) # 创建proxy_data_x
        proxy_data_y.extend(dataset_label[idx_class_proxy]) # 创建proxy_data_y
        test_data_x.extend(dataset_image[idx_class_test]) # 创建test_data_x
        test_data_y.extend(dataset_label[idx_class_test]) # 创建test_data_y
    proxy_data_x = np.array(proxy_data_x) # 创建proxy_data_x
    proxy_data_y = np.array(proxy_data_y) # 创建proxy_data_y
    test_data_x = np.array(test_data_x) # 创建test_data_x
    test_data_y = np.array(test_data_y) # 创建test_data_y

    X_proxy = torch.Tensor(proxy_data_x).type(torch.float32) # 创建X_proxy
    y_proxy = torch.Tensor(proxy_data_y).type(torch.int64) # 创建y_proxy

    data_proxy = [(x, y) for x, y in zip(X_proxy, y_proxy)] # 创建data_proxy
    proxy_loader = DataLoader(data_proxy, batch_size=args.test_batch_size, shuffle=True, drop_last=False) # 创建proxy_loader
    return test_data_x, test_data_y, proxy_loader # 返回test_data_x, test_data_y, proxy_loader


def split_proxy(x, y, args, AT=None): # 分割代理
    client_x, client_at, client_y, proxy_data_x, proxy_data_at, proxy_data_y = [], [], [], [], [], [] # 创建client_x, client_at, client_y, proxy_data_x, proxy_data_at, proxy_data_y

    classes_ls = [i for i in range(args.num_classes)] # 创建classes_ls

    for client in range(args.num_user): # 遍历args.num_user
        dataset_image = x[client] # 创建dataset_image
        dataset_label = y[client] # 创建dataset_label
        idxs = np.array(range(len(dataset_label))) # 创建idxs
        idx_for_each_class = {} # 创建idx_for_each_class
        all_class_x = [] # 创建all_class_x
        all_class_y = [] # 创建all_class_y
        all_class_x_proxy = [] # 创建all_class_x_proxy
        all_class_y_proxy = [] # 创建all_class_y_proxy
        for i in classes_ls: # 遍历classes_ls
            idx_for_each_class[i] = idxs[dataset_label == i] # 创建idx_for_each_class
            num_class_proxy = len(idx_for_each_class[i])*args.proxy_frac # 创建num_class_proxy
            idx_class_proxy = np.random.choice(idx_for_each_class[i], int(num_class_proxy)) # 创建idx_class_proxy
            idx_class_client = list(set(idx_for_each_class[i])-set(idx_class_proxy)) # 创建idx_class_client
            all_class_x_proxy.extend(dataset_image[idx_class_proxy]) # 创建all_class_x_proxy
            all_class_y_proxy.extend(dataset_label[idx_class_proxy]) # 创建all_class_y_proxy
            all_class_x.extend(dataset_image[idx_class_client]) # 创建all_class_x
            all_class_y.extend(dataset_label[idx_class_client]) # 创建all_class_y
        client_x.append(all_class_x) # 创建client_x
        client_y.append(all_class_y) # 创建client_y
        proxy_data_x.append(all_class_x_proxy) # 创建proxy_data_x
        proxy_data_y.append(all_class_y_proxy) # 创建proxy_data_y

    client_loaders, test_loaders = split_data(client_x, client_y, args) # 创建client_loaders, test_loaders
    proxy_client_loaders, proxy_test_loaders = split_data(proxy_data_x, proxy_data_y, args) # 创建proxy_client_loaders, proxy_test_loaders

    return client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders # 返回client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders

def split_data(X, y, args, client_at=None): # 分割数据
    client_loaders, test_loaders = [], [] # 创建client_loaders, test_loaders
    if args.forget_paradigm == 'client': # 如果args.forget_paradigm为client
        train_size = 0.7 # 创建train_size
    else:
        train_size = 0.99 # 创建train_size
    for i in range(len(y)): # 遍历y
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True) # 创建X_train, X_test, y_train, y_test

        train_data = [(x, y) for x, y in zip(X_train, y_train)] # 创建train_data
        test_data = [(x, y) for x, y in zip(X_test, y_test)] # 创建test_data
        client_loaders.append(DataLoader(train_data, batch_size=args.local_batch_size, shuffle=True, num_workers=2, drop_last=False)) # 创建client_loaders
        test_loaders.append(DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, drop_last=False)) # 创建test_loaders

    del X, y # 删除X, y
    return client_loaders, test_loaders # 返回client_loaders, test_loaders


