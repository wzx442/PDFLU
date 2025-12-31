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
    args = get_args()
    set_random_seed(args.seed) 

    shamir = Shamir()

    threshold = int(np.log2(args.num_user))
    print(f"threshold: {threshold}")
    self_seeds, pair_matrix, self_shares, pair_shares = generate_seeds_and_shares(args.num_user, threshold, shamir)
    #endregion

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print('device:', args.device) 

    model = model_init(args) 
    model_copy = copy.deepcopy(model) 

    # data preparation 
    client_all_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = data_init(args) 
    # print(test_loaders[0]) 

    
    

    args.if_unlearning = False 
    case = pdflu_unlearning.PDFLU(args) 

    if args.forget_paradigm == 'client': 
        


        client_all_loaders_process_size = calculate_forget_client_loaders_size(client_all_loaders, args.forget_client_idx)
        print(f"the total size of all data (MB): {client_all_loaders_process_size} MB")

        client_all_loaders_process, test_loaders_process = baizhanting_attack(args, copy.deepcopy(client_all_loaders), copy.deepcopy(test_loaders)) 
        proxy_client_loaders_process, proxy_test_loaders_process = baizhanting_attack(args, copy.deepcopy(proxy_client_loaders), copy.deepcopy(proxy_test_loaders)) 

        model, all_client_models = case.train_normal(model, client_all_loaders_process, test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold)
        
        args.if_unlearning = True 
        if args.paradigm == 'pdflu':
            unlearning_model = case.forget_client_train(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold)    
            
        elif args.paradigm == 'fused':
            unlearning_model = case.forget_client_train(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold)    
        elif args.paradigm == 'retrain':
            unlearning_model = case.FL_Retrain(copy.deepcopy(model_copy), copy.deepcopy(client_all_loaders), test_loaders_process, args) 
        elif args.paradigm == 'federaser':
            unlearning_model = case.federated_learning_unlearning(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders_process, args) 
        elif args.paradigm == 'exactfun':
            unlearning_model = case.federated_unlearning(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
        elif args.paradigm == 'eraseclient':
            unlearning_model = case.fl_unlearning(copy.deepcopy(model), copy.deepcopy(client_all_loaders), test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 


        if args.MIT: 
            args.save_normal_result = False       
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), client_all_loaders_process, test_loaders, proxy_client_loaders_process, proxy_client_loaders, proxy_test_loaders_process, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
            args.save_normal_result = True 
        if args.relearn: 
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_process, test_loaders_process) 
            
    elif args.forget_paradigm == 'none':
        model, all_client_models = case.train_normal(model, client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
    
    elif args.forget_paradigm == 'class': 
        client_all_loaders_bk = copy.deepcopy(client_all_loaders) 
        proxy_client_loaders_bk = copy.deepcopy(proxy_client_loaders) 

        model, all_client_models = case.train_normal(model, copy.deepcopy(client_all_loaders), test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
        
        args.if_unlearning = True 
        for user in range(args.num_user): 
            train_ls = [] 
            proxy_train_ls = [] 
            for data, target in client_all_loaders[user]: 
                data = data.tolist() 
                targets = target.tolist() 
                for idx, label in enumerate(targets): 
                    if label in args.forget_class_idx: 
                        label_ls = [i for i in range(args.num_classes)] 
                        label_ls.remove(label) 
                        inverse_label = np.random.choice(label_ls) 
                        label = inverse_label 
                    train_ls.append((torch.tensor(data[idx]), torch.tensor(label))) 
            for data, target in proxy_client_loaders[user]: 
                data = data.tolist() 
                targets = target.tolist() 
                for idx, label in enumerate(targets): 
                    if label in args.forget_class_idx: 
                        label_ls = [i for i in range(args.num_classes)] 
                        label_ls.remove(label) 
                        inverse_label = np.random.choice(label_ls) 
                        label = inverse_label 
                    proxy_train_ls.append((torch.tensor(data[idx]), torch.tensor(label))) 
            train_loader = DataLoader(train_ls, batch_size=args.test_batch_size, shuffle=True) 
            proxy_train_loader = DataLoader(proxy_train_ls, batch_size=args.test_batch_size, shuffle=True) 
            client_all_loaders[user] = train_loader 
            proxy_client_loaders[user] = proxy_train_loader 
        if args.paradigm == 'pdflu':
            unlearning_model = case.forget_class(copy.deepcopy(model), client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
        elif args.paradigm == 'fused':
            unlearning_model = case.forget_class(copy.deepcopy(model), client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
        elif args.paradigm == 'retrain':
            unlearning_model = case.FL_Retrain(copy.deepcopy(model_copy), copy.deepcopy(client_all_loaders), test_loaders, args) 

        if args.MIT: 
            args.save_normal_result = False 
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), copy.deepcopy(client_all_loaders_bk), test_loaders, proxy_client_loaders_bk, proxy_client_loaders, proxy_test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
            args.save_normal_result = True 
        if args.relearn: 
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_bk, test_loaders) 

    elif args.forget_paradigm == 'sample':
        client_all_loaders_attack = backdoor_attack(args, copy.deepcopy(client_all_loaders)) 
        proxy_client_loaders_attack = backdoor_attack(args, copy.deepcopy(proxy_client_loaders)) 
        model, all_client_models = case.train_normal(model, client_all_loaders_attack, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
        args.if_unlearning = True 
        client_all_loaders_process = erase_backdoor(args, copy.deepcopy(client_all_loaders)) 
        proxy_client_loaders_process = erase_backdoor(args, copy.deepcopy(proxy_client_loaders)) 
        if args.paradigm == 'pdflu':
            unlearning_model = case.forget_sample(copy.deepcopy(model), client_all_loaders_process, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
        elif args.paradigm == 'fused':
            unlearning_model = case.forget_sample(copy.deepcopy(model), client_all_loaders_process, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
        elif args.paradigm == 'retrain':
            unlearning_model = case.FL_Retrain(copy.deepcopy(model_copy), copy.deepcopy(client_all_loaders_process), test_loaders, args) 
        if args.MIT: 
            args.save_normal_result = False 
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), client_all_loaders_attack, test_loaders, proxy_client_loaders_attack, proxy_client_loaders_process, proxy_test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold) 
            args.save_normal_result = True 
        if args.relearn: 
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_attack, test_loaders) 

    else: print("error")
