import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from dataset.data_utils import data_set, separate_data, split_proxy
import numpy as np
import os
import pickle as pkl
import pandas as pd
import tqdm
import random
from transformers import BertTokenizer
from torch.utils.data.sampler import SubsetRandomSampler

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'

# Allocate data to users


def data_init(FL_params): # 初始化数据
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.device == 'cuda' else {} # 创建kwargs
    dataset_x = [] # 创建dataset_x
    dataset_at = [] # 创建dataset_at
    dataset_y = [] # 创建dataset_y

    trainset, testset = data_set(FL_params.data_name) # 创建trainset, testset

    # obtain training indices that will be used for validation
    num_train = len(trainset)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    # split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    # train_idx, valid_idx = indices[split:], indices[:split]
    train_idx = indices[:]

    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)

    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, num_workers=2, **kwargs, drop_last=False) # 创建test_loader
    # train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2, **kwargs, drop_last=False) # 创建train_loader
    train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, num_workers=2, **kwargs, sampler=train_sampler, drop_last=False) # 创建train_loader

    for train_data in train_loader:
        x_train, y_train = train_data # 创建x_train, y_train
        dataset_x.extend(x_train.cpu().detach().numpy()) # 创建dataset_x
        dataset_y.extend(y_train.cpu().detach().numpy()) # 创建dataset_y
    if FL_params.forget_paradigm == 'client':
        for test_data in test_loader: # 遍历test_loader
            x_test, y_test = test_data # 创建x_test, y_test
            dataset_x.extend(x_test.cpu().detach().numpy()) # 创建dataset_x
            dataset_y.extend(y_test.cpu().detach().numpy()) # 创建dataset_y

    dataset_x = np.array(dataset_x) # 创建dataset_x
    dataset_y = np.array(dataset_y) # 创建dataset_y

    X, y, statistic = separate_data((dataset_x, dataset_y), FL_params.num_user, FL_params.num_classes, FL_params, FL_params.niid, FL_params.balance, FL_params.partition, class_per_client=2) # 创建X, y, statistic

    client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = split_proxy(X, y, FL_params) # 创建client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders
    FL_params.datasize_ls = [len(k) for k in X] # 创建FL_params.datasize_ls
    if FL_params.forget_paradigm == 'client': # 如果FL_params.forget_paradigm为client
        test_loaders = test_loaders
        proxy_test_loaders = proxy_test_loaders
    else: # 如果FL_params.forget_paradigm不为client
        proxy_test_x = [] # 创建proxy_test_x
        proxy_test_y = [] # 创建proxy_
        for i in range(FL_params.num_user): # 遍历FL_params.num_user
            for x, y in test_loaders[i]: # 遍历test_loaders[i]
                proxy_test_x.append(x) # 创建proxy_test_x
                proxy_test_y.append(y) # 创建proxy_test_y
        
        
        proxy_test_x = torch.cat(proxy_test_x).numpy() # 创建proxy_test_x
        proxy_test_y = torch.cat(proxy_test_y).numpy() # 创建proxy_test_y
        proxy_test_loader = DataLoader(TensorDataset(torch.tensor(proxy_test_x), torch.tensor(proxy_test_y)), batch_size=FL_params.test_batch_size, shuffle=True, drop_last=False) # 创建proxy_test_loader
        proxy_test_loaders = [proxy_test_loader for _ in range(FL_params.num_user)] # 创建proxy_test_loaders
    return client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders # 返回client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders

def cross_data_init(FL_params): # 初始化交叉数据
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.device == 'cuda' else {} # 创建kwargs
    dataset_x = [] # 创建dataset_x
    dataset_y = [] # 创建dataset_y

    trainset, testset = data_set(FL_params.data_name) # 创建trainset, testset

    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, num_workers=2, **kwargs, drop_last=False) # 创建test_loader
    train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2, **kwargs, drop_last=False) # 创建train_loader

    for train_data in train_loader: # 遍历train_loader
        x_train, y_train = train_data # 创建x_train, y_train
        dataset_x.extend(x_train.cpu().detach().numpy()) # 创建dataset_x
        dataset_y.extend(y_train.cpu().detach().numpy()) # 创建dataset_y
    if FL_params.forget_paradigm == 'client': # 如果FL_params.forget_paradigm为client
        for test_data in test_loader: # 遍历test_loader
            x_test, y_test = test_data # 创建x_test, y_test
            dataset_x.extend(x_test.cpu().detach().numpy()) # 创建dataset_x
            dataset_y.extend(y_test.cpu().detach().numpy()) # 创建dataset_y

    dataset_x = np.array(dataset_x) # 创建dataset_x
    dataset_y = np.array(dataset_y) # 创建dataset_y

    class_num = int(FL_params.num_classes/FL_params.num_user) # 创建class_num
    X = [] # 创建X
    y = [] # 创建y
    idx_ls = [] # 创建idx_ls
    for user in range(FL_params.num_user): # 遍历FL_params.num_user
        idx = [] # 创建idx
        for i in range(class_num): # 遍历class_num
            item = user*class_num + i # 创建item
            indices = [idx for idx, label in enumerate(dataset_y) if label == item] # 创建indices
            idx.extend(indices) # 创建idx
        idx_ls.append(idx) # 创建idx_ls
    corss_idx = idx_ls[0][:int(len(idx_ls[0])*0.01)] # 创建corss_idx
    idx_ls[0] = idx_ls[0][int(len(idx_ls[0])*0.01):] # 创建idx_ls
    idx_ls[1] = corss_idx + idx_ls[1] # 创建idx_ls
    remain_idx = [] # 创建remain_idx
    for idx in range(1, FL_params.num_user): # 遍历FL_params.num_user
        remain_idx.extend(idx_ls[idx]) # 创建remain_idx
    random.shuffle(remain_idx) # 随机打乱remain_idx
    sublist_size = len(remain_idx) // (FL_params.num_user-len(FL_params.forget_client_idx)) # 创建sublist_size
    remainder = len(remain_idx) % (FL_params.num_user-len(FL_params.forget_client_idx)) # 创建remainder

    sublists = [remain_idx[i * sublist_size + min(i, remainder):(i + 1) * sublist_size + min(i + 1, remainder)] for i in
                range(9)] # 创建sublists

    for idx in range(1, FL_params.num_user): # 遍历FL_params.num_user
        idx_ls[idx] = sublists[idx-1] # 创建idx_ls

    for user in range(FL_params.num_user): # 遍历FL_params.num_user
        X.append(dataset_x[idx_ls[user]]) # 创建X
        y.append(dataset_y[idx_ls[user]]) # 创建y

    for i in range(FL_params.num_user): # 遍历FL_params.num_user
        print('client {} data size {} lable {}'.format(i, len(X[i]),np.unique(y[i]))) # 打印客户端数据大小

    client_loaders, test_loaders, proxy_loader = split_proxy(X, y, FL_params) # 创建client_loaders, test_loaders, proxy_loader
    FL_params.datasize_ls = [len(k) for k in X] # 创建FL_params.datasize_ls
    if FL_params.forget_paradigm == 'client': # 如果FL_params.forget_paradigm为client
        test_loaders = test_loaders # 创建test_loaders
    else: # 如果FL_params.forget_paradigm不为client 
        test_loaders = [test_loader for _ in range(FL_params.num_user)] # 创建test_loaders

    return client_loaders, test_loaders, proxy_loader # 返回client_loaders, test_loaders, proxy_loader
