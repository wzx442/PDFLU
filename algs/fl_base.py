import sys

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
from utilss.utils import *
import os
import pandas as pd
import logging
from alg_utils.ada_hessian import AdaHessian
from transformers import AdamW
from utilss.init_enc import *

class Base(object):
    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        self.log_dir = f"logs/retrain_{self.args.forget_paradigm}_{self.args.data_name}_{self.args.alpha}"
        


   

    def FL_Retrain(self, init_global_model,  client_all_loaders, test_loaders, FL_params):
        # if (FL_params.if_retrain == False): 
            # raise ValueError('FL_params.if_retrain should be set to True, if you want to retrain FL model') 

        print('\n')
        print(5 * "#" + "   Federated Retraining Start  " + 5 * "#")
        std_time = time.time() 
        # retrain_GMs = list()

        # retrain_GMs.append(copy.deepcopy(init_global_model))
        global_model = copy.deepcopy(init_global_model) 
        checkpoints_ls = [] 
        # gap = 0
        result_list = [] 
        avg_list = [] 
        for epoch in range(FL_params.global_epoch): 
            # last_gap = gap
            selected_clients = list(np.random.choice(range(FL_params.num_user), size=int(FL_params.num_user * FL_params.fraction), replace=False)) 
            if FL_params.forget_paradigm == 'client': 
                selected_clients = [value for value in selected_clients if value not in FL_params.forget_client_idx] 
            self.select_forget_idx = list() 
            # select_client_loaders = list()
            record = -1 
            for idx in selected_clients: 
                # select_client_loaders.append(client_all_loaders[idx])
                record += 1 
                if idx in self.args.forget_client_idx: 
                    self.select_forget_idx.append(record) 
            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients) 

            client_models, client_data_num = self.global_train_once(epoch, global_model, selected_clients, select_client_loaders, test_loaders, FL_params, checkpoints_ls) 
            if client_models:  
                global_model = self.fedavg(client_models) 
            else:
                print("warnning: client_models is null")
            # global_model_ls = [copy.deepcopy(global_model) for _ in range(FL_params.num_user)]
            if self.args.forget_paradigm == 'client':  
        
                avg_f_acc,         avg_r_acc,     test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders) 
                avg_train_f_acc, avg_train_r_acc = train_client_group_accuracy(self, global_model, self.args, client_all_loaders)
                print('Epoch {}, Remember Test Acc={}, Forget Test Acc={}, avg_train_f_acc={}, avg_train_r_acc={}'.format(epoch, avg_r_acc, avg_f_acc, avg_train_f_acc, avg_train_r_acc)) 
                avg_list.append((epoch, avg_r_acc, avg_f_acc, avg_train_f_acc, avg_train_r_acc))

            elif self.args.forget_paradigm == 'class': 
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args, test_loaders) 
                avg_train_f_acc, avg_train_r_acc = train_class_group_accuracy(self, global_model, self.args, client_all_loaders)
                print('Epoch {}, Remember Test Acc={}, Forget Test Acc={}, avg_train_f_acc={}, avg_train_r_acc={}'.format(epoch, avg_r_acc, avg_f_acc, avg_train_f_acc, avg_train_r_acc)) 
                avg_list.append((epoch, avg_r_acc, avg_f_acc, avg_train_f_acc, avg_train_r_acc))
            elif self.args.forget_paradigm == 'sample': # if sample unleaning
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model, self.args, test_loaders) 
                print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero, avg_test_acc)) 
                avg_list.append((epoch, avg_jingdu, avg_acc_zero, avg_test_acc))
            result_list.extend(test_result_ls)

            # gap = avg_r_acc - avg_f_acc

            # retrain_GMs.append(copy.deepcopy(global_model))

        end_time = time.time() # end time
        consume_time = end_time - std_time # consume time
        if FL_params.forget_paradigm == 'client': # if forget paradigm is client
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss']) # create dataframe
            df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_r_acc', 'Avg_f_acc', 'Avg_train_f_acc', 'Avg_train_r_acc']) # create dataframe
        elif FL_params.forget_paradigm == 'class': # if forget paradigm is class
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss']) # create dataframe
            df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_r_acc', 'Avg_f_acc', 'Avg_train_f_acc', 'Avg_train_r_acc']) # create dataframe
        elif FL_params.forget_paradigm == 'sample': # if forget paradigm is sample
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc']) # create dataframe
            df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_jingdu', 'Avg_acc_zero', 'Avg_test_acc']) # create dataframe
        df['Comsume_time'] = consume_time # add consume time
        df_avg['Comsume_time'] = consume_time # add consume time

        if self.args.cut_sample == 1.0: # if cut sample is 1.0
            if self.args.save_normal_result: # if save normal result
                df.to_csv(
                './results/retrain/{}/Acc_loss_retrain_{}_data_{}_distri_{}_fnum_{}_{}.csv'.format(
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.data_name, 
                                                                                        self.args.alpha, 
                                                                                        len(self.args.forget_class_idx),
                                                                                        self.args.file_name
                                                                                        ))
                df_avg.to_csv(
                './logs/retrain/{}/Acc_loss_retrain_{}_data_{}_distri_{}_fnum_{}_{}.csv'.format(
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.data_name, 
                                                                                        self.args.alpha, 
                                                                                        len(self.args.forget_class_idx),
                                                                                        self.args.file_name
                                                                                        ))
        elif self.args.cut_sample < 1.0: # if cut sample is less than 1.0
            if self.args.save_normal_result: # if save normal result
                df.to_csv(
                    './results/retrain/{}/Acc_loss_retrain_{}_data_{}_distri_{}_fnum_{}_partdata_{}_{}.csv'.format(
                                                                                        self.args.forget_paradigm,
                                                                                        self.args.forget_paradigm,
                                                                                        self.args.data_name,
                                                                                        self.args.alpha,
                                                                                        len(self.args.forget_class_idx), 
                                                                                        self.args.cut_sample,
                                                                                        self.args.file_name
                                                                                        ))
                df_avg.to_csv(
                './logs/retrain/{}/Acc_loss_retrain_{}_data_{}_distri_{}_fnum_{}_partdata_{}_{}.csv'.format(
                                                                                        self.args.forget_paradigm,
                                                                                        self.args.forget_paradigm,
                                                                                        self.args.data_name,
                                                                                        self.args.alpha,
                                                                                        len(self.args.forget_class_idx),
                                                                                        self.args.cut_sample,
                                                                                        self.args.file_name
                                                                                        ))

        print(5 * "#" + "  Federated Retraining End  " + 5 * "#")
        return global_model



    # Function:
    # For the global round of training, the data and optimizer of each global_ModelT is used. The global model of the previous round is the initial point and the training begins.
    # NOTE:The global model inputed is the global model for the previous round
    #     The output client_Models is the model that each user trained separately.
    def global_train_once(
            self, # self
            epoch, 
            global_model, 
            selected_clients, 
            client_data_loaders, 
            test_loaders, 
            FL_params, 
            checkpoints_ls, 
            ) -> list[nn.Module, int]:
        """
        
        Function:
            For the global round of training, the data and optimizer of each global_ModelT is used. The global model of the previous round is the initial point and the training begins.
        Note: The global model inputed is the global model for the previous round
        Output: The model that each client trained separately and their indices idx, as well as the amount of data for each client
        """
        global_model.to(FL_params.device) # Move the global model to the device
        device_cpu = torch.device("cpu") # Create CPU device
        client_models = [] # List of client models
        client_data_num = [] # List of client data amounts
        lr = FL_params.lr # Learning rate

        # if FL_params.paradigm == 'federaser':
        #     for ii in range(len(client_data_loaders)):
        #         client_models.append("1")
        # else:
        # for ii in range(len(client_data_loaders)):
        #     client_models.append(copy.deepcopy(global_model))

        print(f"Number of clients participating in training: {len(selected_clients)}")
        for idx, client_data in enumerate(client_data_loaders): # Iterate over client data
            # print(f"Client {selected_clients[idx]} data amount: {len(client_data.dataset)}")
            client_data_num.append(len(client_data.dataset))

            model = copy.deepcopy(global_model) # Copy the global model
            if self.args.data_name == 'cifar10':
                if epoch % 10 == 0 and epoch > 0:
                    lr = lr * 0.5
                # Optimize SGD parameters for CIFAR-10, remove nesterov for stability
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

            # model.to(device)
            model.train() # Train the model

            # local training
            model = self.local_train(model, optimizer, client_data_loaders[idx], FL_params) # Local training
            

            client_models.append(model) # Add client model
            # print(f"Client {selected_clients[idx]} local training completed")

            # if self.args.paradigm == 'lora': # If paradigm is lora
            if self.args.paradigm == 'pdflu': # If paradigm is pdflu
                for name, param in model.named_parameters(): # Iterate over model parameters
                    for name_, param_ in global_model.named_parameters(): # Iterate over global model parameters
                        if name == name_: # If parameter names are the same
                            pdist = nn.PairwiseDistance(p=1) # Create PairwiseDistance
                            param_size = sys.getsizeof(param.data) # Get parameter size
                            diff = pdist(param.data, param_.data) # Calculate parameter difference
                            diff = torch.norm(diff) # Calculate norm of parameter difference
                            self.param_change_dict[name] = diff # Add parameter difference
                            self.param_size[name] = param_size # Add parameter size
            model.to(device_cpu) # Move model to CPU
            # Get client data amount
            # client_data_num.append(FL_params.datasize_ls[selected_clients[idx]]) # Add client data amount
        return client_models, client_data_num # Return client models and client indices


    """
    Function:
    Test the performance of the model on the test set
    """
    def local_train(self, model, optimizer, data_loader, FL_params):
        """
        Function:
            Test the performance of the model on the test set
        """
        # Use a milder learning rate scheduler
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=FL_params.local_epoch//2, gamma=0.9)
        
        for local_epoch in range(FL_params.local_epoch): # Iterate over local epochs
            criteria = nn.CrossEntropyLoss().to(FL_params.device) # Create cross-entropy loss function
            for batch_idx, (data, target) in enumerate(data_loader):  # Iterate over data loader
                optimizer.zero_grad() # Clear gradients
                data = data.to(FL_params.device) # Move data to device
                target = target.to(FL_params.device) # Move target to device
                pred = model(data) # Model output
                pred = pred.to(FL_params.device) # Move prediction to device

                loss = criteria(pred, target) # Calculate loss
                loss.backward() # Calculate gradients
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step() # Update parameters
            
            # Update learning rate
            # scheduler.step()

        # 
        # model.eval()
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for data, target in data_loader:
        #         data = data.to(FL_params.device)
        #         target = target.to(FL_params.device)
        #         output = model(data)
        #         pred = output.argmax(dim=1)
        #         correct += (pred == target).sum().item()
        #         total += target.size(0)
        # train_acc = correct / total if total > 0 else 0.0
        # model.train()
        return model

    def evaluate_training_acc(self, model, client_all_loaders, FL_params):
        model.eval()
        acc_list = []
        with torch.no_grad():
            for loader in client_all_loaders:
                correct = 0
                total = 0
                for data, target in loader:
                    data = data.to(FL_params.device)
                    target = target.to(FL_params.device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                acc = correct / total if total > 0 else 0.0
                acc_list.append(acc)
        model.train()
        avg_acc = sum(acc_list) / len(acc_list)
        return avg_acc, acc_list


    def local_train_infocom22(self, model, optimizer, data_loader, FL_params): # Local training infocom22
        optimizer = AdaHessian(model.parameters()) # Create AdaHessian optimizer
        for local_epoch in range(FL_params.local_epoch): # Iterate over local epochs
            for batch_idx, (data, target) in enumerate(data_loader): # Iterate over data loader
                model.zero_grad() # Clear gradients
                data = data.to(FL_params.device) # Move data to device
                target = target.to(FL_params.device) # Move target to device

                pred, _ = model(data) # Model output
                criteria = nn.CrossEntropyLoss() # Create cross-entropy loss function
                loss = criteria(pred, target) # Calculate loss
                loss.backward(create_graph=True) # Calculate gradients
                optimizer.step() # Update parameters
                optimizer.zero_grad() # Clear gradients


    def test(self, model, test_loader, FL_params): # Test
        # for param in model.parameters(): # Iterate over model parameters
            # device = param.device # Get device
            # break
        model.eval() # Evaluation mode
        test_loss = 0 # Test loss
        correct = 0 # Correct count
        total = 0 # Total count
        true_zero_total = 0 # True zero total
        acc_true_zero_total = 0 # True zero accuracy
        pred_zero_total = 0 # Predicted zero total
        jingdu_pred_zero_total = 0 # Predicted zero precision
        # criteria = nn.CrossEntropyLoss() # Create cross-entropy loss function
        criteria = nn.CrossEntropyLoss().to(FL_params.device)
        
        # Ensure model is on the correct device
        model.to(FL_params.device)
        
        with torch.no_grad(): # Do not calculate gradients
            for data, target in test_loader: # Iterate over test data loader
                data = data.to(FL_params.device) # Move data to device
                target = target.to(FL_params.device) # Move target to device
                output = model(data) # Model output
                test_loss += criteria(output, target)  # sum up batch loss Calculate loss
                pred = torch.argmax(output, axis=1) # Prediction
                pred = pred.cpu() # Move prediction to CPU
                target = target.cpu() # Move target to CPU
                if FL_params.forget_paradigm == 'sample':
                    pred_zero_indices = np.where(pred == 0) # Predicted zero indices
                    pred_zero_count = np.count_nonzero(pred == 0) # Predicted zero count
                    pred_zero_total += pred_zero_count # Predicted zero total
                    for ele in target[pred_zero_indices]:
                        if ele == 0:
                            jingdu_pred_zero_total += 1 # Predicted zero precision
                    true_zero_indices = np.where(target == 0) # True zero indices
                    for ele1 in pred[true_zero_indices]: # Iterate over true zero indices
                        if ele1 == 0:
                            acc_true_zero_total += 1 # True zero accuracy
                    true_zero_count = np.count_nonzero(target == 0) # True zero count
                    true_zero_total += true_zero_count # True zero total
                    correct += torch.sum(torch.eq(pred, target)).item() # Calculate correct count
                    total += len(target) # Calculate total count
                else:
                    correct += torch.sum(torch.eq(pred, target)).item() # Calculate correct count
                    total += len(target) # Calculate total count
        if FL_params.forget_paradigm == 'sample': # If forget paradigm is sample
            if pred_zero_total == 0: # If predicted zero total is 0
                jingdu = np.nan # Precision is NaN
                print("pred_zero_total == 0: Predicted zero total is 0")
            else:
                jingdu = jingdu_pred_zero_total/pred_zero_total # Calculate precision
            if true_zero_total == 0: # If true zero total is 0
                acc_zero = np.nan # Accuracy is NaN
                print("true_zero_total == 0: True zero total is 0")
            else:
                acc_zero = acc_true_zero_total/true_zero_total # Calculate accuracy
            test_acc = correct/total # Calculate test accuracy
            return (jingdu, acc_zero, test_acc) # Return precision, accuracy, test accuracy
        else:
            test_loss /= len(test_loader.dataset) # Calculate test loss
            test_acc = correct/total # Calculate test accuracy
            return (test_loss, test_acc) # Return test loss, test accuracy


    # def test_normal(self, model, test_loader, FL_params): # Test normal model accuracy and loss
    #     model.eval() # Evaluation mode
    #     test_loss = 0 # Test loss
    #     correct = 0 # Correct count
    #     total = 0 # Total count
    #     criteria = nn.CrossEntropyLoss().to(FL_params.device) # Create cross-entropy loss function
    #     with torch.no_grad(): # Do not calculate gradients
    #         for data, target in test_loader: # Iterate over test data loader
    #             data = data.to(FL_params.device) # Move data to device
    #             target = target.to(FL_params.device) # Move target to device
    #             model.to(FL_params.device) # Move model to device
    #             output = model(data) # Model output
    #             test_loss += criteria(output, target)  # sum up batch loss Calculate loss
    #             pred = torch.argmax(output, axis=1) # Prediction
    #             pred = pred.cpu() # Move prediction to CPU
    #             target = target.cpu() # Move target to CPU
    #             correct += torch.sum(torch.eq(pred, target)).item() # Calculate correct count
    #             total += len(target) # Calculate total count
    #     test_loss /= len(test_loader.dataset) # Calculate test loss
    #     test_acc = correct/total # Calculate test accuracy
    #     return (test_loss, test_acc) # Return test loss, test accuracy

    # 
    # Function:
    # FedAvg
    #
    #     Parameters
    #     ----------
    #     local_models : list of local models 
    #         DESCRIPTION.In federated learning, with the global_model as the initial model, each user uses a collection of local models updated with their local data.
    #         
    #     local_model_weights : tensor or array 
    #         DESCRIPTION. The weight of each local model is usually related to the accuracy rate and number of data of the local model.(Bypass)
    #       

    #     Returns
    #     -------
    #     update_global_model Updated global model
    #         Updated global model using fedavg algorithm
    #         Using fedavg algorithm to update global model
    #     """
    def fedavg(self, local_models):
        """
        Input:
        local_models : list of local models
            In federated learning, with the global_model as the initial model, each user uses a collection of local models updated with their local data.
        local_model_weights : local model weights
            DESCRIPTION. The weight of each local model is usually related to the accuracy rate and number of data of the local model.(Bypass)

        Output:
        Updated global model
        Using fedavg algorithm to update global model
        """
        # print("\nStarting federated averaging")
        global_model = copy.deepcopy(local_models[0]) # Copy global model: Create an exact copy of the first local model as the initial global model.

        avg_state_dict = global_model.state_dict() # Global model state dictionary: Get all parameters of the current global model (in dictionary form) for subsequent parameter aggregation and updating.
        local_state_dicts = list() # List of local model state dictionaries
        for model in local_models: # Iterate over local models
            local_state_dicts.append(model.state_dict()) # Add local model state dictionary

        for layer in avg_state_dict.keys(): # Iterate over global model state dictionary
            avg_state_dict[layer] = 0 # Initialize global model state dictionary
            # for client_idx in range(len(local_models)):
            #     avg_state_dict[layer] += local_state_dicts[client_idx][layer]*self.args.datasize_ls[client_idx]
            # if 'num_batches_tracked' in layer:
            #     avg_state_dict[layer] = (avg_state_dict[layer]/sum(self.args.datasize_ls)).long()
            # else:
            #     avg_state_dict[layer] /= sum(self.args.datasize_ls)
            for client_idx in range(len(local_models)): # Iterate over local models
                avg_state_dict[layer] += local_state_dicts[client_idx][layer] # Add local model state dictionary
            if 'num_batches_tracked' in layer: # If 'num_batches_tracked' in layer
                avg_state_dict[layer] = (avg_state_dict[layer] / len(local_models)).long() # Update global model state dictionary   
            else:
                avg_state_dict[layer] /= len(local_models) # Update global model state dictionary

        global_model.load_state_dict(avg_state_dict) # Load global model state dictionary

        return global_model


    def weighted_aggregation(self, local_models, client_data_sizes):
        """
        Implement weighted aggregation in federated learning (Federated Averaging).

        Parameters:
        - local_models (List[OrderedDict]): A list containing local model parameters (state_dict) returned from each client
        - client_data_sizes (List[int]): A list containing the amount of training data for each client corresponding to the order of local_models

        Returns:
        - global_model (OrderedDict): Aggregated global model parameters (state_dict)
        """
        print("\nStarting weighted aggregation")
        # for i in range(len(local_models)):
            # print(f"客户端{selected_clients[i]}的数据量: {client_data_sizes[i]}")
        total_data_size = sum(client_data_sizes) # Calculate total data size
        global_model = copy.deepcopy(local_models[0]) # Copy global model: Create an exact copy of the first local model as the initial global model.

        avg_state_dict = global_model.state_dict() # Global model state dictionary: Get all parameters of the current global model (in dictionary form) for subsequent parameter aggregation and updating.
        local_state_dicts = list() # List of local model state dictionaries
        for model in local_models: # Iterate over local models
            local_state_dicts.append(model.state_dict()) # Add local model state dictionary

        for layer in avg_state_dict.keys(): # Iterate over global model state dictionary
            avg_state_dict[layer] = 0 # Initialize global model state dictionary
            for client_idx in range(len(local_models)):
                avg_state_dict[layer] += (local_state_dicts[client_idx][layer] * client_data_sizes[client_idx])
            if 'num_batches_tracked' in layer:
                avg_state_dict[layer] = (avg_state_dict[layer]/total_data_size).long()
            else:
                avg_state_dict[layer] /= total_data_size


        global_model.load_state_dict(avg_state_dict) # Load global model state dictionary
        return global_model

    def sec_weighted_aggregation(self, local_models, client_data_sizes, selected_clients, self_seeds, pair_matrix, self_shares, pair_shares, threshold, FL_params):
        """
        Implement secure weighted aggregation in federated learning (Federated Averaging).

        Parameters:
        - local_models (List[OrderedDict]): A list containing local encrypted model parameters (state_dict) returned from each client
        - client_data_sizes (List[int]): A list containing the amount of training data for each client corresponding to the order of local_models
        - selected_clients (List[int]): A list containing the indices of selected clients, i.e., the IDs of online clients
        - self_seeds (List[int]): A list containing the self-mask seeds for each client corresponding to the order of local_models
        - pair_matrix (List[List[int]]): A list containing the pairwise mask seed matrix for each client corresponding to the order of local_models
        - self_shares (List[int]): A list containing the self-mask shares for each client corresponding to the order of local_models
        - pair_shares (List[List[int]]): A list containing the pairwise mask shares for each client corresponding to the order of local_models
        - threshold (int): An integer representing the recovery threshold corresponding to the order of local_models
        - FL_params (Namespace): A Namespace containing the federated learning parameters

        Returns:
        - global_model (OrderedDict): Aggregated plaintext global model parameters (state_dict)
        """
        print("\nStarting secure weighted aggregation")
        
        online_clients = len(selected_clients) # Calculate the current number of clients

        if online_clients < threshold:
            raise ValueError(f"Number of online clients is less than the recovery threshold: {online_clients} < {threshold}")

        
        # Recover the right encrypted values of offline clients, which are the double masks
        
        shamir_handler = Shamir() # Create an instance of the Shamir class
        
        recover_drop_client_pair_seed = recover_drop(selected_clients, FL_params.num_user, threshold, shamir_handler, pair_matrix, pair_shares) # Sum of pairwise masks that have not been canceled out due to offline clients

        all_online_self_masks = sum(self_seeds[i] for i in selected_clients) # Calculate the sum of self-masks for all online clients

        online_data_size = sum(client_data_sizes) # Calculate the total amount of data for online clients

        global_model = copy.deepcopy(local_models[0]) # Copy the global model: create an exact copy of the first local model as the initial global model

        avg_state_dict = global_model.state_dict() # Global model state dictionary: get all parameters of the current global model (in dictionary form) for subsequent parameter aggregation and updating
        local_state_dicts = list() # List of local model state dictionaries
        for model in local_models: # Iterate over local models
            local_state_dicts.append(model.state_dict()) # Add local model state dictionary

        for layer in avg_state_dict.keys(): # Iterate over global model state dictionary
            avg_state_dict[layer] = 0 # Initialize global model state dictionary
            for client_idx in range(len(local_models)):# Sum parameters at the same index of online clients, weighted by data size
                avg_state_dict[layer] += local_state_dicts[client_idx][layer] 
            avg_state_dict[layer] += recover_drop_client_pair_seed # Recover the right encrypted values of offline clients
            avg_state_dict[layer] -= all_online_self_masks # Subtract the sum of self-masks of online clients

            # Calculate weighted average
            if 'num_batches_tracked' in layer:
                avg_state_dict[layer] = (avg_state_dict[layer]/online_data_size).long()
            else:
                avg_state_dict[layer] /= online_data_size


        global_model.load_state_dict(avg_state_dict) # Load global model state dictionary
        return global_model
    
    def weighted_aggregation_in_relearn(self, local_models, client_data_sizes):
        """
        Implement weighted aggregation for federated learning (Federated Averaging).

        Parameters:
        - local_models (List[OrderedDict]): A list containing local model parameters (state_dict) returned from each client
        - client_data_sizes (List[int]): A list containing the amount of data used for training by each client, corresponding in order to local_models
        Returns:
        - global_model (OrderedDict): Aggregated plaintext global model parameters (state_dict)
        """
        print("\n run weighted aggregation")
        # for i in range(len(local_models)):
        #     print(f"Client {selected_clients[i]} data size: {client_data_sizes[i]}")
        total_data_size = sum(client_data_sizes) # Calculate total data size
        global_model = copy.deepcopy(local_models[0]) # Copy global model: create an exact copy of the first local model as the initial global model.
        avg_state_dict = global_model.state_dict() # Global model state dictionary: get all parameters of the current global model (in dictionary form) for subsequent parameter aggregation and updating.
        local_state_dicts = list() # List of local model state dictionaries
        for model in local_models: # Iterate over local models
            local_state_dicts.append(model.state_dict()) # Append local model state dictionary

        for layer in avg_state_dict.keys(): # Iterate over global model state dictionary
            avg_state_dict[layer] = 0 # Initialize global model state dictionary
            for client_idx in range(len(local_models)):
                avg_state_dict[layer] += (local_state_dicts[client_idx][layer]*client_data_sizes[client_idx])
            if 'num_batches_tracked' in layer:
                avg_state_dict[layer] = (avg_state_dict[layer]/total_data_size).long()
            else:
                avg_state_dict[layer] /= total_data_size


        global_model.load_state_dict(avg_state_dict) # Load global model state dictionary
        return global_model

    def relearn_unlearning_knowledge(self, unlearning_model, client_all_loaders, test_loaders):
        # This function seems to be redundant
        checkpoints_ls = [] # List of checkpoints
        all_global_models = list() # List of all global models
        all_client_models = list() # List of all client models
        global_model = unlearning_model # Global model
        result_list = [] # List of results

        all_global_models.append(global_model) # Append global model
        std_time = time.time() # Standard time
        for epoch in range(self.args.global_epoch): # Iterate over global epochs
            if self.args.forget_paradigm == 'client': # If forget paradigm is client
                select_client_loaders = list() # List of selected client loaders
                for idx in self.args.forget_client_idx: # Iterate over forget client indices
                    select_client_loaders.append(client_all_loaders[idx]) # Append client loader

            elif self.args.forget_paradigm == 'class': # If forget paradigm is class
                select_client_loaders = list() # List of selected client loaders
                client_loaders = select_forget_class(self.args, copy.deepcopy(client_all_loaders)) # Select forget class
                for v in client_loaders: # Iterate over client loaders
                    if v is not None: # If client loader is not None
                        select_client_loaders.append(v) # Append client loader

            elif self.args.forget_paradigm == 'sample': # If forget paradigm is sample
                select_client_loaders = list() # List of selected client loaders
                client_loaders = select_forget_sample(self.args, copy.deepcopy(client_all_loaders)) # Select forget sample
                for v in client_loaders: # Iterate over client loaders
                    if v is not None: # If client loader is not None
                        select_client_loaders.append(v) # Append client loader

            client_models, client_data_num = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls) # Global training once

            all_client_models += client_models # Add client models
            global_model = self.fedavg(client_models) # Federated averaging
            all_global_models.append(copy.deepcopy(global_model).to('cpu')) # Add global model
            end_time = time.time() # End time

            consume_time = end_time - std_time # Consume time

            if self.args.forget_paradigm == 'client': # If forget paradigm is client
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders) # Test client forget
                for item in test_result_ls: # Iterate over test results
                    item.append(consume_time) # Append consume time
                result_list.extend(test_result_ls) # Extend result list
                df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss', 'Comsume_time']) # Create dataframe

            elif self.args.forget_paradigm == 'class': # If forget paradigm is class
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args, test_loaders) # Test class forget
                for item in test_result_ls: # Iterate over test results
                    item.append(consume_time) # Append consume time
                result_list.extend(test_result_ls) # Extend result list
                df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss', 'Comsume_time']) # Create dataframe

            elif self.args.forget_paradigm == 'sample': # If forget paradigm is sample
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model, self.args, test_loaders) # Test backdoor forget
                for item in test_result_ls: # Iterate over test results
                    item.append(consume_time) # Append consume time
                result_list.extend(test_result_ls) # Extend result list
                df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc', 'Comsume_time']) # Create dataframe

            global_model.to('cpu') # Move global model to CPU

            print("Relearn Round = {}".format(epoch)) # Print relearn round
        if self.args.cut_sample == 1.0: # If cut sample is 1.0
            df.to_csv('./results/{}/relearn/relearn_data_{}_distri_{}_fnum_{}_algo_{}_{}.csv'.format(   
                                                                                            self.args.forget_paradigm,
                                                                                            self.args.data_name,
                                                                                            self.args.alpha,
                                                                                            len(self.args.forget_class_idx),
                                                                                            self.args.paradigm,
                                                                                            self.args.file_name), 
                                                                                            index=False)
        elif self.args.cut_sample < 1.0: # If cut sample is less than 1.0
            df.to_csv('./results/{}/relearn/relearn_data_{}_distri_{}_fnum_{}_algo_{}_partdata_{}_{}.csv'.format(    
                                                                                                            self.args.forget_paradigm,
                                                                                                            self.args.data_name,
                                                                                                            self.args.alpha,
                                                                                                            len(self.args.forget_class_idx),
                                                                                                            self.args.paradigm, 
                                                                                                            self.args.cut_sample,
                                                                                                            self.args.file_name), 
                                                                                                            index=False) # Save results 
        return 

    def global_train_once_in_relearn(
            self, # self
            epoch, #
            global_model, 
            client_data_loaders, 
            test_loaders, 
            FL_params, 
            checkpoints_ls, 
            relearn_client_ls, 
            ) -> list[nn.Module, int, int]:
        """

        Function:
            For global round training, use each global model's data and optimizer. The previous round's global model is used as the starting point to begin training.
        Note: The input global model is the global model from the previous round
        Output: Each client's trained model along with their index idx, and the amount of data for each client
        """
        global_model.to(FL_params.device) # Move global model to device
        device_cpu = torch.device("cpu") # Create CPU device
        client_models = [] # Client model list
        client_idx = [] # Client index list
        client_data_num = [] # Client data amount list
        lr = FL_params.lr # Learning rate

        # if FL_params.paradigm == 'federaser':
        #     for ii in range(len(client_data_loaders)):
        #         client_models.append("1")
        # else:
        # for ii in range(len(client_data_loaders)):
        #     client_models.append(copy.deepcopy(global_model))

        for idx, client_data in enumerate(client_data_loaders): # Iterate over client data
            client_data_num.append(len(client_data.dataset))
            # if self.args.forget_paradigm == 'client':
                # print(f"Client {relearn_client_ls[idx]} data amount: {client_data_num[idx]}")
            # else:
                # print(f"Client {idx} data amount: {client_data_num[idx]}")
            model = copy.deepcopy(global_model) # Copy global model

            if self.args.data_name == 'cifar10':
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) # CIFAR-10 uses smaller weight_decay
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) # Create SGD optimizer
            # model.to(device)
            model.train() # Train model

            # local training
            if self.args.paradigm == 'infocom22' and self.args.if_unlearning == True: # If paradigm is infocom22 and unlearning is needed
                self.local_train_infocom22(model, optimizer, client_data_loaders[idx], FL_params) # Local training
            else: # If paradigm is not infocom22 and unlearning is not needed
                model = self.local_train(model, optimizer, client_data_loaders[idx], FL_params) # Local training
            

            client_models.append(model) # Add client model

            if self.args.paradigm == 'lora': # If paradigm is lora
                for name, param in model.named_parameters(): # Iterate over model parameters
                    for name_, param_ in global_model.named_parameters(): # Iterate over global model parameters
                        if name == name_: # If parameter names are the same
                            pdist = nn.PairwiseDistance(p=1) # Create PairwiseDistance
                            param_size = sys.getsizeof(param.data) # Get parameter size
                            diff = pdist(param.data, param_.data) # Calculate parameter difference
                            diff = torch.norm(diff) # Calculate norm of parameter difference
                            self.param_change_dict[name] = diff # Add parameter difference
                            self.param_size[name] = param_size # Add parameter size
            model.to(device_cpu) # Move model to CPU
        return client_models, client_data_num # Return client models and client indices    
    
    def global_train_once_enc(
            self, # self
            epoch, 
            global_model, 
            selected_clients, 
            client_data_loaders,
            test_loaders,
            FL_params, 
            checkpoints_ls,
            self_seeds,
            pair_matrix, 
            self_shares, 
            pair_shares, 
            ) -> list[nn.Module, nn.Module, int]:
        """
        Function:
            For global round training, use the data and optimizer of each global model. The global model from the previous round is used as the starting point to begin training.
                Note: The input global model is the global model from the previous round
        Output: The plaintext models trained by each client and their encrypted models, as well as the amount of data for each client
        """
        # print("\nStarting global training round {}".format(epoch))
        global_model.to(FL_params.device) 
        device_cpu = torch.device("cpu") 
        client_models = [] 
        enc_client_models = [] 
        client_data_num = [] 
        lr = FL_params.lr 



        print(f"Number of participating clients: {len(selected_clients)}")
        for idx, client_data in enumerate(client_data_loaders): # Iterate over client data
            # Get client data size
            data_size_idx = len(client_data.dataset)
            client_data_num.append(data_size_idx) # Add client data size

            # print(f"Client {selected_clients[idx]} data size: {data_size_idx}")

            model = copy.deepcopy(global_model) # Copy global model

            # if self.args.data_name == 'cifar10':
            #     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) # CIFAR-10 uses smaller weight_decay
            # else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) # Create SGD optimizer
            # model.to(device)
            model.train() # Train model

            # local training
            if self.args.paradigm == 'infocom22' and self.args.if_unlearning == True: # If paradigm is infocom22 and unlearning is needed
                self.local_train_infocom22(model, optimizer, client_data_loaders[idx], FL_params) # Local training
            else: # If paradigm is not infocom22 and unlearning is not needed
                model = self.local_train(model, optimizer, client_data_loaders[idx], FL_params) # Local training
            
            client_models.append(model) # Add client model
            # Encryption
            enc_model = copy.deepcopy(model) # Copy model

            enc_model_i = self.encrypt_model(
                                            enc_model, 
                                            selected_clients[idx], 
                                            len(selected_clients), 
                                            data_size_idx, 
                                            self_seeds, 
                                            pair_matrix) # Encryption

            enc_client_models.append(enc_model_i) # Add encrypted client model
            
 
            if self.args.paradigm == 'pdflu': 
                for name, param in model.named_parameters():
                    for name_, param_ in global_model.named_parameters(): 
                        if name == name_: 
                            pdist = nn.PairwiseDistance(p=1)
                            param_size = sys.getsizeof(param.data) 
                            diff = pdist(param.data, param_.data) 
                            diff = torch.norm(diff) 
                            self.param_change_dict[name] = diff 
                            self.param_size[name] = param_size 
            model.to(device_cpu) 
            enc_model.to(device_cpu) 
            
        return client_models, enc_client_models, client_data_num
    
    def encrypt_model(self, model, i,n, data_size_i, self_seeds, pair_matrix):

        X = self_seeds[i] + sum(pair_matrix[i][j] for j in range(i + 1, n)) - sum(pair_matrix[i][j] for j in range(i))

        for name, param in model.named_parameters(): 
            param.data *= data_size_i 
            param.data += X 

        return model
    
    def encrypt_model_sample(self, model, data_size_i):


        for name, param in model.named_parameters(): 
            param.data *= data_size_i 

        return model
