import copy

from dataset.generate_data import data_init, cross_data_init
import torch

from algs import pdflu_unlearning, fl_base
from utilss.utils import *
import random
import numpy as np

from utilss.args import get_args
from utilss.init_enc import *




def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # hyperparameters setting 超参数设置    
    args = get_args()
    set_random_seed(args.seed) # 设置随机种子

    #region: 初始化种子
    # 创建 Shamir 处理器实例
    shamir = Shamir()

    # 生成种子和份额
    # 秘密共享阈值为log2(args.num_user)
    threshold = int(np.log2(args.num_user))
    print(f"秘密共享阈值为: {threshold}")
    self_seeds, pair_matrix, self_shares, pair_shares = generate_seeds_and_shares(args.num_user, threshold, shamir)
    #endregion

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设置设备
    print('device:', args.device) # 打印设备

    model = model_init(args) # 初始化模型
    model_copy = copy.deepcopy(model) # 复制模型

    # data preparation 数据准备
    client_all_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = data_init(args) # 初始化数据
    # print(test_loaders[0]) # 打印测试数据

    # 横坐标表示客户端编号，纵坐标表示label，网格里面填充该label的样本数
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # from collections import Counter

    # # 创建一个矩阵，行表示label，列表示客户端
    # num_classes = args.num_classes
    # num_clients = args.num_user
    # data_matrix = np.zeros((num_classes, num_clients), dtype=int)
    
    # # 遍历所有客户端，统计每个客户端每个标签的样本数
    # for client_id in range(num_clients):
    #     client_loader = client_all_loaders[client_id]
    #     labels = []
    #     for _, target in client_loader:
    #         labels.extend(target.numpy())
        
    #     label_counts = Counter(labels)
    #     for label, count in label_counts.items():
    #         data_matrix[label, client_id] = count

    # # 保存data_matrix为CSV文件
    # import pandas as pd
    # df = pd.DataFrame(data_matrix, 
    #                  columns=[f'Client_{i}' for i in range(num_clients)],
    #                  index=[f'Label_{i}' for i in range(num_classes)])
    # df.to_csv("clients_data_matrix_alpha=03.csv")
    # print("数据矩阵已保存到 clients_data_matrix_alpha=03.csv")
    
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(data_matrix, annot=True, fmt="d", cmap="viridis", 
    #             xticklabels=range(num_clients), yticklabels=range(num_classes))
    # plt.xlabel("客户端编号 (Client ID)")
    # plt.ylabel("标签 (Label)")
    # plt.title("所有客户端的训练数据分布 (Training Data Distribution Across All Clients)")
    # plt.savefig("clients_data_distribution_alpha=03.png", dpi=300, bbox_inches='tight')
    # print("客户端数据分布图已保存到 clients_data_distribution_alpha=03.png")
    
    # exit()



    args.if_unlearning = False # 是否进行遗忘学习
    case = pdflu_unlearning.PDFLU(args) # 创建PDFLU类实例

    if args.forget_paradigm == 'client': # 遗忘学习方式为客户端遗忘
        # 计算遗忘客户端的数据量大小所占的空间大小（MB）



        # 遗忘客户端的数据量大小所占的空间大小（MB）
        client_all_loaders_process_size = calculate_forget_client_loaders_size(client_all_loaders, args.forget_client_idx)
        print(f"遗忘客户端的数据量大小所占的空间大小 (MB): {client_all_loaders_process_size} MB")

        client_all_loaders_process, test_loaders_process = baizhanting_attack(args, copy.deepcopy(client_all_loaders), copy.deepcopy(test_loaders)) # 白盒攻击
        proxy_client_loaders_process, proxy_test_loaders_process = baizhanting_attack(args, copy.deepcopy(proxy_client_loaders), copy.deepcopy(proxy_test_loaders)) # 白盒攻击

        # model, all_client_models = case.train_normal(model, client_all_loaders_process, test_loaders_process) # 原始代码 训练正常模型
        model, all_client_models = case.train_normal(model, client_all_loaders_process, test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 训练正常模型, 用于遗忘学习
        # model, all_client_models = case.train_normal(model, client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 训练正常模型, 用于遗忘学习
    
        args.if_unlearning = True # 设置为遗忘学习模式
        if args.paradigm == 'pdflu':
            # 因为只有idx的客户端的数据被投毒，所以客户端遗忘的情况下，遗忘idx后的client_all_loaders与client_all_loaders_process相同
            # print("=== 执行PDFLU客户端遗忘学习 ===")
            unlearning_model = case.forget_client_train(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 遗忘学习模型   
            # unlearning_model = case.forget_client_train(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 遗忘学习模型   

        elif args.paradigm == 'fused':
            unlearning_model = case.forget_client_train(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 遗忘学习模型   
        elif args.paradigm == 'retrain':
            unlearning_model = case.FL_Retrain(copy.deepcopy(model_copy), copy.deepcopy(client_all_loaders), test_loaders_process, args) # 遗忘学习模型
        elif args.paradigm == 'federaser':
            unlearning_model = case.federated_learning_unlearning(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders_process, args) # 遗忘学习模型
        elif args.paradigm == 'exactfun':
            unlearning_model = case.federated_unlearning(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 遗忘学习模型
        elif args.paradigm == 'eraseclient':
            unlearning_model = case.fl_unlearning(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 遗忘学习模型


        # print("=== 遗忘学习完成，准备执行成员推理攻击 ===")
        # print(f"args.MIT = {args.MIT}")
        if args.MIT: # 使用成员推理攻击
            # print("=== 开始执行成员推理攻击 ===")
            args.save_normal_result = False # 不保存正常模型的结果      
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), client_all_loaders_process, test_loaders, proxy_client_loaders_process, proxy_client_loaders, proxy_test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 成员推理攻击
            args.save_normal_result = True # 保存正常模型的结果
        if args.relearn: # 重新学习遗忘的知识
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_process, test_loaders_process) # 重新学习遗忘的知识
            
    elif args.forget_paradigm == 'none':
        # 利用未被投毒的数据进行训练
        model, all_client_models = case.train_normal(model, client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 训练正常模型
    
    elif args.forget_paradigm == 'class': # 遗忘学习方式为类别遗忘
        client_all_loaders_bk = copy.deepcopy(client_all_loaders) # 备份客户端数据
        proxy_client_loaders_bk = copy.deepcopy(proxy_client_loaders) # 备份代理客户端数据

        model, all_client_models = case.train_normal(model, copy.deepcopy(client_all_loaders), test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 训练正常模型
        
        args.if_unlearning = True # 设置为遗忘学习模式
        for user in range(args.num_user): # 遍历所有客户端
            train_ls = [] # 训练数据
            proxy_train_ls = [] # 代理训练数据
            for data, target in client_all_loaders[user]: # 遍历客户端数据
                data = data.tolist() # 数据
                targets = target.tolist() # 目标标签
                for idx, label in enumerate(targets): # 遍历目标标签
                    if label in args.forget_class_idx: # 如果目标标签在遗忘类别中
                        label_ls = [i for i in range(args.num_classes)] # 标签列表
                        label_ls.remove(label) # 移除目标标签
                        inverse_label = np.random.choice(label_ls) # 随机选择一个标签
                        label = inverse_label # 替换目标标签
                    train_ls.append((torch.tensor(data[idx]), torch.tensor(label))) # 添加训练数据
            for data, target in proxy_client_loaders[user]: # 遍历代理客户端数据
                data = data.tolist() # 数据
                targets = target.tolist() # 目标标签
                for idx, label in enumerate(targets): # 遍历目标标签
                    if label in args.forget_class_idx: # 如果目标标签在遗忘类别中
                        label_ls = [i for i in range(args.num_classes)] # 标签列表
                        label_ls.remove(label) # 移除目标标签
                        inverse_label = np.random.choice(label_ls) # 随机选择一个标签
                        label = inverse_label # 替换目标标签
                    proxy_train_ls.append((torch.tensor(data[idx]), torch.tensor(label))) # 添加代理训练数据
            train_loader = DataLoader(train_ls, batch_size=args.test_batch_size, shuffle=True) # 训练数据
            proxy_train_loader = DataLoader(proxy_train_ls, batch_size=args.test_batch_size, shuffle=True) # 代理训练数据
            client_all_loaders[user] = train_loader # 更新客户端数据
            proxy_client_loaders[user] = proxy_train_loader # 更新代理客户端数据
        if args.paradigm == 'pdflu':
            unlearning_model = case.forget_class(copy.deepcopy(model), client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 创建遗忘学习模型
        elif args.paradigm == 'fused':
            unlearning_model = case.forget_class(copy.deepcopy(model), client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 创建遗忘学习模型
        elif args.paradigm == 'retrain':
            unlearning_model = case.FL_Retrain(copy.deepcopy(model_copy), copy.deepcopy(client_all_loaders), test_loaders, args) # 遗忘学习模型

        # print("=== 类别遗忘完成，准备执行成员推理攻击 ===")
        # print(f"args.MIT = {args.MIT}")
        if args.MIT: # 使用成员推理攻击
            # print("=== 开始执行成员推理攻击 (类别遗忘) ===")
            args.save_normal_result = False # 不保存正常模型的结果
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), copy.deepcopy(client_all_loaders_bk), test_loaders, proxy_client_loaders_bk, proxy_client_loaders, proxy_test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 成员推理攻击
            args.save_normal_result = True # 保存正常模型的结果
        if args.relearn: # 重新学习遗忘的知识
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_bk, test_loaders) # 重新学习遗忘的知识

    elif args.forget_paradigm == 'sample': # 遗忘学习方式为样本遗忘
        client_all_loaders_attack = backdoor_attack(args, copy.deepcopy(client_all_loaders)) # 添加后门
        proxy_client_loaders_attack = backdoor_attack(args, copy.deepcopy(proxy_client_loaders)) # 添加后门
        model, all_client_models = case.train_normal(model, client_all_loaders_attack, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 训练正常模型
        args.if_unlearning = True # 设置为遗忘学习模式
        client_all_loaders_process = erase_backdoor(args, copy.deepcopy(client_all_loaders)) # 删除后门
        proxy_client_loaders_process = erase_backdoor(args, copy.deepcopy(proxy_client_loaders)) # 删除后门
        if args.paradigm == 'pdflu':
            unlearning_model = case.forget_sample(copy.deepcopy(model), client_all_loaders_process, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 遗忘学习模型
        elif args.paradigm == 'fused':
            unlearning_model = case.forget_sample(copy.deepcopy(model), client_all_loaders_process, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 遗忘学习模型
        elif args.paradigm == 'retrain':
            unlearning_model = case.FL_Retrain(copy.deepcopy(model_copy), copy.deepcopy(client_all_loaders_process), test_loaders, args) # 遗忘学习模型

        # print("=== 样本遗忘完成，准备执行成员推理攻击 ===")
        # print(f"args.MIT = {args.MIT}")
        if args.MIT: # 使用成员推理攻击
            # print("=== 开始执行成员推理攻击 (样本遗忘) ===")
            args.save_normal_result = False # 不保存正常模型的结果
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), client_all_loaders_attack, test_loaders, proxy_client_loaders_attack, proxy_client_loaders_process, proxy_test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) # 成员推理攻击
            args.save_normal_result = True # 保存正常模型的结果
        if args.relearn: # 重新学习遗忘的知识
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_attack, test_loaders) # 重新学习遗忘的知识

    else: print("遗忘学习方式错误")
