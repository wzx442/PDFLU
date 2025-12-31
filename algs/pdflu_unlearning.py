import time
import math
import pandas as pd
import torch

from models.Model_base import *
from models import LeNet_FashionMNIST, CNN_Cifar10, CNN_Cifar100, Model_base, ViT_Cifar100
from utilss.utils import *
from utilss.calculate_model_size import calculate_model_size, calculate_model_size_specific
from dataset.data_utils import *
from algs.fl_base import Base
import torch.optim as optim
import copy
import logging
import matplotlib.pyplot as plt

import random
from models.Model_base import *

from utilss.e_c_utils import Utils
from utilss.fusion import Fusion, FusionAvg, FusionRetrain


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")



class PDFLU(Base):
    def __init__(self, args):
        super(PDFLU, self).__init__(args)
        self.args = args
        self.log_dir = f"logs/pdflu_{self.args.data_name}_{self.args.alpha}"
        self.param_change_dict = {}
        self.param_size = {}

    def train_normal(self, global_model, client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold):

        checkpoints_ls = [] 
        result_list = [] 
        param_list = [] 
        avg_list = [] 
        global_model_size = 0


        model_size = calculate_model_size(global_model)
        print(f"size of global_model: {model_size} MB")


        for name, param in global_model.named_parameters(): 
            # print(name)
            self.param_change_dict[name] = 0 
            self.param_size[name] = 0 

        for epoch in range(self.args.global_epoch): 
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user*self.args.fraction), replace=False)) 
            select_client_loaders = [client_all_loaders[idx] for idx in selected_clients]
            if self.args.paradigm == 'pdflu':
                client_models, enc_client_models, client_data_num = self.global_train_once_enc(
                                                                                                epoch=epoch, 
                                                                                                global_model=global_model,
                                                                                                selected_clients=selected_clients, 
                                                                                                client_data_loaders=select_client_loaders, 
                                                                                                test_loaders=test_loaders, 
                                                                                                FL_params=self.args, 
                                                                                                checkpoints_ls=checkpoints_ls,
                                                                                                self_seeds=self_seeds,
                                                                                                pair_matrix=pair_matrix,
                                                                                                self_shares=self_shares,
                                                                                                pair_shares=pair_shares) 
                # global_model = self.sec_weighted_aggregation(enc_client_models, client_data_num, selected_clients, self_seeds, pair_matrix, self_shares, pair_shares, threshold, self.args) 
                global_model = self.weighted_aggregation(client_models, client_data_num) 

            else:
                client_models, client_data_num = self.global_train_once(
                                                                        epoch=epoch, 
                                                                        global_model=global_model, 
                                                                        selected_clients=selected_clients, 
                                                                        client_data_loaders=select_client_loaders, 
                                                                        test_loaders=test_loaders, 
                                                                        FL_params=self.args, 
                                                                        checkpoints_ls=checkpoints_ls) 
                global_model = self.fedavg(client_models) 

            





            if self.args.forget_paradigm == 'sample': 
                
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model, self.args, test_loaders) 
                print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero, avg_test_acc)) 
                result_list.extend(test_result_ls) 
                avg_list.append((epoch, avg_jingdu, avg_acc_zero, avg_test_acc, model_size)) 
            
            elif self.args.forget_paradigm == 'client': 
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders) 
                
                avg_train_f_acc, avg_train_r_acc = train_client_group_accuracy(self, global_model, self.args, client_all_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}, avg_train_f_acc={}, avg_train_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc, avg_train_f_acc, avg_train_r_acc))
                result_list.extend(test_result_ls) 
                avg_list.append((epoch, avg_f_acc, avg_r_acc, avg_train_f_acc, avg_train_r_acc, model_size)) 
            
            elif self.args.forget_paradigm == 'class': 
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args, test_loaders) 
                
                avg_train_f_acc, avg_train_r_acc = train_class_group_accuracy(self, global_model, self.args, client_all_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}, avg_train_f_acc={}, avg_train_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc, avg_train_f_acc, avg_train_r_acc))
                result_list.extend(test_result_ls) 
                avg_list.append((epoch, avg_f_acc, avg_r_acc, avg_train_f_acc, avg_train_r_acc, model_size)) 
            elif self.args.forget_paradigm == 'none':
                
                

                (test_loss, test_acc) = test_normal(self, global_model, test_loaders, self.args, selected_clients) 
                avg_train_acc, train_acc_list = self.evaluate_training_acc(global_model, select_client_loaders, self.args)
                print('Epoch={}, test_loss={}, test_acc={}, avg_train_acc={}'.format(epoch, test_loss, test_acc, avg_train_acc)) 
                result_list.append((epoch, test_loss, test_acc, avg_train_acc)) 
                avg_list.append((epoch, test_loss, test_acc, avg_train_acc, model_size)) 

                

            if self.args.paradigm == 'pdflu': 
                diff_ls = list(self.param_change_dict.values()) 
                name = list(self.param_change_dict.keys()) 
                # print("name")
                # print(type(name),name)
                diff_ls_ = [float(i) for i in diff_ls]  
                
                param_list.append(diff_ls_) 
                # diff_ls_.append(list(self.param_size.values()))
        # print("param_list")
        # print(type(param_list),param_list)
        if self.args.paradigm == 'pdflu':
            df = pd.DataFrame(param_list, columns=name) 
            df.to_csv('./results/param_change_{}_distri_{}_{}.csv'.format(self.args.data_name, self.args.alpha, self.args.file_name)) 

        # print("save global model")
        torch.save(global_model.state_dict(), 'save_model/global_model_{}_{}.pth'.format(self.args.data_name, self.args.file_name)) 

        if self.args.forget_paradigm == 'sample': 
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc']) 
            df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_jingdu', 'Avg_acc_zero', 'Avg_test_acc', 'Model_size']) 
        elif self.args.forget_paradigm == 'client': 
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss']) 
            df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_f_acc', 'Avg_r_acc', 'Avg_train_f_acc', 'Avg_train_r_acc', 'Model_size']) 
        elif self.args.forget_paradigm == 'class': 
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss']) 
            df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_f_acc', 'Avg_r_acc', 'Avg_train_f_acc', 'Avg_train_r_acc', 'Model_size']) 
        elif self.args.forget_paradigm == 'none': 
            df = pd.DataFrame(result_list, columns=['Epoch', 'Test_loss', 'Test_acc', 'Avg_train_acc']) 
            df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Test_loss', 'Test_acc', 'Avg_train_acc', 'Model_size']) 
        if self.args.save_normal_result: 
            df.to_csv('./results/{}/FL/Acc_loss_fl_{}_data_{}_distri_{}_{}.csv'.format(
                                                                                self.args.forget_paradigm, 
                                                                                self.args.forget_paradigm, 
                                                                                self.args.data_name, 
                                                                                self.args.alpha, 
                                                                                self.args.file_name
                                                                                )) 
            df_avg.to_csv('./logs/{}/FL/Acc_loss_fl_{}_data_{}_distri_{}_{}.csv'.format(
                                                                                self.args.forget_paradigm, 
                                                                                self.args.forget_paradigm, 
                                                                                self.args.data_name, 
                                                                                self.args.alpha, 
                                                                                self.args.file_name
                                                                                )) 
        return global_model, client_models 

    def forget_client_train(self, global_model, client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold):
        
        global_model.load_state_dict(torch.load('save_model/global_model_{}_{}.pth'.format(self.args.data_name, self.args.file_name))) 
        avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, 1, global_model, self.args, test_loaders) 
        avg_train_f_acc, avg_train_r_acc = train_client_group_accuracy(self, global_model, self.args, client_all_loaders)
        print('before unlearning: client-level forget, Avg_r_acc: {}, Avg_f_acc: {}, avg_train_f_acc: {}, avg_train_r_acc: {}'.format(avg_r_acc, avg_f_acc, avg_train_f_acc, avg_train_r_acc)) 

        pdflu_model = Lora(self.args, global_model, self.args.a) 
        torch.save(pdflu_model.state_dict(), 'save_model/global_pdflu_model_{}_{}.pth'.format(self.args.data_name, self.args.file_name)) 

        model_size = calculate_model_size_specific(global_model, self.args)
        print(f"size of pdflu_model: {model_size} MB")

        checkpoints_ls = []
        result_list = []
        avg_list = []
        consume_time = 0 

        print("\n=====start forget client training, total {} rounds=====".format(self.args.FU_epoch))
        for epoch in range(self.args.FU_epoch): 
            pdflu_model.train() 
            selected_clients = [i for i in range(self.args.num_user) if i not in self.args.forget_client_idx] 
            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients) 


            std_time = time.time() 
            if self.args.paradigm == 'pdflu':
                client_models, enc_client_models, client_data_num = self.global_train_once_enc(epoch=epoch, 
                                                                                               global_model=pdflu_model, 
                                                                                               selected_clients=selected_clients, 
                                                                                               client_data_loaders=select_client_loaders, 
                                                                                               test_loaders=test_loaders, 
                                                                                               FL_params=self.args, 
                                                                                               checkpoints_ls=checkpoints_ls,
                                                                                               self_seeds=self_seeds,
                                                                                               pair_matrix=pair_matrix,
                                                                                               self_shares=self_shares,
                                                                                               pair_shares=pair_shares) 
                
            else:
                client_models, client_data_num = self.global_train_once(epoch=epoch, 
                                                                        global_model=pdflu_model, 
                                                                        selected_clients=selected_clients, 
                                                                        client_data_loaders=select_client_loaders, 
                                                                        test_loaders=test_loaders, 
                                                                        FL_params=self.args, 
                                                                        checkpoints_ls=checkpoints_ls) 

            end_time = time.time() 

            if self.args.paradigm == 'pdflu':
                # avg_model = self.sec_weighted_aggregation(enc_client_models, client_data_num, selected_clients, self_seeds, pair_matrix, self_shares, pair_shares, threshold, self.args) 
                avg_model = self.weighted_aggregation(client_models, client_data_num)        

            else:
                avg_model = self.fedavg(client_models) 

            consume_time += end_time - std_time 

            pdflu_model.load_state_dict(avg_model.state_dict())

            pdflu_model.eval() 

            avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, pdflu_model, self.args, test_loaders) 
            avg_train_f_acc, avg_train_r_acc = train_client_group_accuracy(self, pdflu_model, self.args, client_all_loaders)
            
            result_list.extend(test_result_ls) 

            avg_list.append((epoch, avg_f_acc, avg_r_acc, avg_train_f_acc, avg_train_r_acc, model_size)) 

            print(f'PDFLU: epoch={epoch}, client-level forget, avg_r_acc={avg_r_acc}, avg_f_acc={avg_f_acc}, avg_train_r_acc={avg_train_r_acc}, avg_train_f_acc={avg_train_f_acc}') 


        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss']) 
        df['Comsume_time'] = consume_time   
        df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_f_acc', 'Avg_r_acc', 'Avg_train_f_acc', 'Avg_train_r_acc', 'Model_size']) 
        df_avg['Comsume_time'] = consume_time   
        if self.args.cut_sample == 1.0: 
            if self.args.save_normal_result: 
                df.to_csv('./results/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_{}.csv'.format(self.args.forget_paradigm,
                                                                                                      self.args.forget_paradigm,
                                                                                                      self.args.data_name,
                                                                                                      self.args.alpha,
                                                                                                      len(self.args.forget_class_idx),
                                                                                                      self.args.file_name
                                                                                                      ))
                df_avg.to_csv('./logs/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_{}.csv'.format(self.args.forget_paradigm,
                                                                                                       self.args.forget_paradigm,
                                                                                                       self.args.data_name,
                                                                                                       self.args.alpha,
                                                                                                       len(self.args.forget_class_idx),
                                                                                                       self.args.file_name
                                                                                                       ))
        elif self.args.cut_sample < 1.0: 
            if self.args.save_normal_result: 
                df.to_csv('./results/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_partdata_{}_{}.csv'.format(
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.data_name,
                                                                                                        self.args.alpha,
                                                                                                        len(self.args.forget_class_idx), 
                                                                                                        self.args.cut_sample,
                                                                                                        self.args.file_name
                                                                                                        ))
                df_avg.to_csv('./logs/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_partdata_{}_{}.csv'.format(
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.data_name,
                                                                                                        self.args.alpha,
                                                                                                        len(self.args.forget_class_idx), 
                                                                                                        self.args.cut_sample,
                                                                                                        self.args.file_name
                                                                                                        ))

        return pdflu_model 

    def forget_class(self, global_model, client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold): 
        checkpoints_ls = [] 
        result_list = [] 
        avg_list = [] 
        consume_time = 0 

        model_size = calculate_model_size_specific(global_model, self.args)

        pdflu_model = Lora(self.args, global_model, self.args.a) 
        print(f"\n======star sample unlearning{self.args.FU_epoch}======")
        for epoch in range(self.args.FU_epoch): 
            pdflu_model.train() 
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False)) 

            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients) 
            std_time = time.time() 

            if self.args.paradigm == 'pdflu':
                client_models, enc_client_models, client_data_num = self.global_train_once_enc(
                                                                                            epoch=epoch, 
                                                                                            global_model=pdflu_model, 
                                                                                            selected_clients=selected_clients, 
                                                                                            client_data_loaders=select_client_loaders, 
                                                                                            test_loaders=test_loaders, 
                                                                                            FL_params=self.args, 
                                                                                            checkpoints_ls=checkpoints_ls,
                                                                                            self_seeds=self_seeds,
                                                                                            pair_matrix=pair_matrix,
                                                                                            self_shares=self_shares,
                                                                                            pair_shares=pair_shares) 
                forget_model_size = calculate_model_size_specific(enc_client_models[0], self.args)
                print(f"size of global model: {forget_model_size} MB")
            else:
                client_models, client_data_num = self.global_train_once(epoch, pdflu_model, selected_clients, select_client_loaders, test_loaders, self.args, checkpoints_ls) 

            end_time = time.time() 
            if self.args.paradigm == 'pdflu':
                # pdflu_model = self.sec_weighted_aggregation(enc_client_models, client_data_num, selected_clients, self_seeds, pair_matrix, self_shares, pair_shares, threshold, self.args) 
                pdflu_model = self.weighted_aggregation(client_models, client_data_num)           

            else:
                pdflu_model = self.fedavg(client_models) 
            consume_time += end_time-std_time 
            avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, pdflu_model, self.args, test_loaders) 
            avg_train_f_acc, avg_train_r_acc = train_class_group_accuracy(self, pdflu_model, self.args, client_all_loaders)
            result_list.extend(test_result_ls) 
            avg_list.append((epoch, avg_f_acc, avg_r_acc, avg_train_f_acc, avg_train_r_acc, model_size)) 
            print(f'PDFLU: epoch={epoch}, class-level forget, avg_r_acc={avg_r_acc}, avg_f_acc={avg_f_acc}, avg_train_r_acc={avg_train_r_acc}, avg_train_f_acc={avg_train_f_acc}') 

        
        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss']) 
        df['Comsume_time'] = consume_time 
        df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_f_acc', 'Avg_r_acc', 'Avg_train_f_acc', 'Avg_train_r_acc', 'Model_size']) 
        df_avg['Comsume_time'] = consume_time 
        if self.args.cut_sample == 1.0: 
            if self.args.save_normal_result: 
                df.to_csv('./results/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_{}.csv'.format(
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.data_name, 
                                                                                        self.args.alpha, 
                                                                                        len(self.args.forget_class_idx),
                                                                                        self.args.file_name
                                                                                        )) 
                df_avg.to_csv('./logs/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_{}.csv'.format(
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.data_name, 
                                                                                        self.args.alpha, 
                                                                                        len(self.args.forget_class_idx),
                                                                                        self.args.file_name
                                                                                        )) 
        elif self.args.cut_sample < 1.0: 
            if self.args.save_normal_result: 
                df.to_csv('./results/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_partdata_{}_{}.csv'.format(  
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.data_name,
                                                                                                        self.args.alpha,
                                                                                                        len(self.args.forget_class_idx), 
                                                                                                        self.args.cut_sample,
                                                                                                        self.args.file_name
                                                                                                        )) 
                df_avg.to_csv('./logs/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_partdata_{}_{}.csv'.format(
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.data_name,
                                                                                                        self.args.alpha,
                                                                                                        len(self.args.forget_class_idx), 
                                                                                                        self.args.cut_sample,
                                                                                                        self.args.file_name
                                                                                                        )) 

        return pdflu_model 

    def forget_sample(self, global_model, client_all_loaders, test_loaders, self_seeds, pair_matrix, self_shares, pair_shares, threshold): 
        checkpoints_ls = [] 
        result_list = [] 
        avg_list = [] 
        consume_time = 0 

        model_size = calculate_model_size_specific(global_model, self.args)

        pdflu_model = Lora(self.args, global_model, self.args.a) 
        print(f"\n======star sample unlearning{self.args.FU_epoch}======")
        for epoch in range(self.args.FU_epoch): # 
            pdflu_model.train() 
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False)) 

            self.select_forget_idx = list() 
            select_client_loaders = list() 
            record = -1 
            for idx in selected_clients: 
                select_client_loaders.append(client_all_loaders[idx]) 
                record += 1 
                if idx in self.args.forget_client_idx: 
                    self.select_forget_idx.append(record) 
            std_time = time.time() 
            if self.args.paradigm == 'pdflu':
                client_models, enc_client_models, client_data_num = self.global_train_once_enc(
                                                                                epoch=epoch, 
                                                                                global_model=pdflu_model, 
                                                                                selected_clients=selected_clients, 
                                                                                client_data_loaders=select_client_loaders, 
                                                                                test_loaders=test_loaders, 
                                                                                FL_params=self.args, 
                                                                                checkpoints_ls=checkpoints_ls,
                                                                                self_seeds=self_seeds,
                                                                                pair_matrix=pair_matrix,
                                                                                self_shares=self_shares,
                                                                                pair_shares=pair_shares) 
                
            else:
                client_models, client_data_num = self.global_train_once(epoch, pdflu_model, selected_clients, select_client_loaders, test_loaders, self.args, checkpoints_ls) 

            end_time = time.time() 
            if self.args.paradigm == 'pdflu':
                # pdflu_model = self.sec_weighted_aggregation(enc_client_models, client_data_num, selected_clients, self_seeds, pair_matrix, self_shares, pair_shares, threshold, self.args) 
                pdflu_model = self.weighted_aggregation(client_models, client_data_num)         
                
            else:
                pdflu_model = self.fedavg(client_models)  
            # pdflu_model = self.weighted_aggregation(client_models, client_data_num, selected_clients) 
            # pdflu_model = self.sec_weighted_aggregation(enc_client_models, client_data_num, selected_clients, self_seeds, pair_matrix, self_shares, pair_shares, threshold, self.args) 
            consume_time += end_time-std_time 

            avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, pdflu_model, self.args, test_loaders) 
            result_list.extend(test_result_ls) 
            avg_list.append((epoch, avg_jingdu, avg_acc_zero, avg_test_acc, model_size)) 
            print(f'PDFLU: epoch={epoch}, sample-level forget, avg_jingdu={avg_jingdu}, avg_acc_zero={avg_acc_zero}, avg_test_acc={avg_test_acc}') 

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc']) 
        df['Comsume_time'] = consume_time 
        df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_jingdu', 'Avg_acc_zero', 'Avg_test_acc', 'Model_size']) 
        df_avg['Comsume_time'] = consume_time 
        if self.args.cut_sample == 1.0: 
            if self.args.save_normal_result: 
                df.to_csv('./results/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_{}.csv'.format(
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.data_name, 
                                                                                        self.args.alpha, 
                                                                                        len(self.args.forget_class_idx), 
                                                                                        self.args.file_name
                                                                                        )) 
                df_avg.to_csv('./logs/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_{}.csv'.format(
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.forget_paradigm, 
                                                                                        self.args.data_name, 
                                                                                        self.args.alpha, 
                                                                                        len(self.args.forget_class_idx), 
                                                                                        self.args.file_name
                                                                                        )) 
        elif self.args.cut_sample < 1.0: 
            if self.args.save_normal_result: 
                df.to_csv('./results/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_partdata_{}_{}.csv'.format(
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.data_name,
                                                                                                        self.args.alpha, 
                                                                                                        len(self.args.forget_class_idx), 
                                                                                                        self.args.cut_sample,
                                                                                                        self.args.file_name
                                                                                                        )) 
                df_avg.to_csv('./logs/{}/FU/Acc_loss_pdflu_{}_data_{}_distri_{}_fnum_{}_partdata_{}_{}.csv'.format(
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.data_name,
                                                                                                        self.args.alpha, 
                                                                                                        len(self.args.forget_class_idx), 
                                                                                                        self.args.cut_sample,
                                                                                                        self.args.file_name
                                                                                                        )) 

        return pdflu_model 

    def relearn_unlearning_knowledge(self, unlearning_model, client_all_loaders, test_loaders): 
        print("\star unlerning, total{}epoch".format(self.args.global_epoch))
        checkpoints_ls = [] 
        all_global_models = list() 
        all_client_models = list() 
        global_model = unlearning_model 
        result_list = [] 
        avg_list = [] 
        relearn_client_ls =[]

        all_global_models.append(global_model)
        std_time = time.time()
        for epoch in range(self.args.global_epoch): 
            print(f"\nRelearn Round = {epoch}") 

            if self.args.forget_paradigm == 'client': 
                select_client_loaders = list() 
                for idx in self.args.forget_client_idx: 
                    select_client_loaders.append(client_all_loaders[idx]) 
                    relearn_client_ls.append(idx)
                print(f"retrain_client_ls: {relearn_client_ls}")
            
            elif self.args.forget_paradigm == 'class': 
                select_client_loaders = list()
                client_loaders = select_forget_class(self.args, copy.deepcopy(client_all_loaders)) 
                for v in client_loaders: 
                    if v is not None: 
                        select_client_loaders.append(v)
            elif self.args.forget_paradigm == 'sample': 
                select_client_loaders = list() 
                client_loaders = select_forget_sample(self.args, copy.deepcopy(client_all_loaders)) 
                for v in client_loaders: 
                    if v is not None: 
                        select_client_loaders.append(v)

            client_models, client_data_num = self.global_train_once_in_relearn(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls, relearn_client_ls) 

            all_client_models += client_models 
            if self.args.paradigm == 'pdflu':
                global_model = self.weighted_aggregation_in_relearn(client_models, client_data_num) 
                
                
            else:
                global_model = self.fedavg(client_models) 

            all_global_models.append(copy.deepcopy(global_model).to('cpu')) 
            end_time = time.time() 

            consume_time = end_time - std_time 

            if self.args.forget_paradigm == 'client':     
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders) 
                print(f"avg_f_acc: {avg_f_acc}, avg_r_acc: {avg_r_acc}")
                for item in test_result_ls: 
                    item.append(consume_time) 
                result_list.extend(test_result_ls) 
                avg_list.append((epoch, avg_f_acc, avg_r_acc)) 
                df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss', 'Comsume_time']) 
                df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_f_acc', 'Avg_r_acc']) 
                df_avg['Comsume_time'] = consume_time 

            elif self.args.forget_paradigm == 'class': 
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args, test_loaders) 
                print(f"avg_f_acc: {avg_f_acc}, avg_r_acc: {avg_r_acc}")
                for item in test_result_ls: 
                    item.append(consume_time) 
                result_list.extend(test_result_ls) 
                avg_list.append((epoch, avg_f_acc, avg_r_acc)) 
                df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss', 'Comsume_time'])  
                df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_f_acc', 'Avg_r_acc']) 
                df_avg['Comsume_time'] = consume_time 

            elif self.args.forget_paradigm == 'sample': 
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model, self.args, test_loaders) 
                print(f"avg_jingdu: {avg_jingdu}, avg_acc_zero: {avg_acc_zero}, avg_test_acc: {avg_test_acc}")
                for item in test_result_ls: 
                    item.append(consume_time) 
                result_list.extend(test_result_ls) 
                avg_list.append((epoch, avg_jingdu, avg_acc_zero, avg_test_acc)) 
                df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc', 'Comsume_time']) 
                df_avg = pd.DataFrame(avg_list, columns=['Epoch', 'Avg_jingdu', 'Avg_acc_zero', 'Avg_test_acc']) 
                df_avg['Comsume_time'] = consume_time 

            global_model.to('cpu') 

           
        
        if self.args.cut_sample == 1.0: 
            df.to_csv('./results/{}/relearn/relearn_data_{}_distri_{}_fnum_{}_algo_{}_{}.csv'.format(
                                                                                            self.args.forget_paradigm,
                                                                                            self.args.data_name,
                                                                                            self.args.alpha,
                                                                                            len(self.args.forget_class_idx),
                                                                                            self.args.paradigm,
                                                                                            self.args.file_name), 
                                                                                            index=False) 
            df_avg.to_csv('./logs/{}/relearn/relearn_data_{}_distri_{}_fnum_{}_algo_{}_{}.csv'.format(
                                                                                            self.args.forget_paradigm,
                                                                                            self.args.data_name,
                                                                                            self.args.alpha,
                                                                                            len(self.args.forget_class_idx),
                                                                                            self.args.paradigm,
                                                                                            self.args.file_name), index=False) 
        elif self.args.cut_sample < 1.0: 
            df.to_csv('./results/{}/relearn/relearn_data_{}_distri_{}_fnum_{}_algo_{}_partdata_{}_{}.csv'.format(  
                                                                                                        self.args.forget_paradigm,
                                                                                                        self.args.data_name,
                                                                                                        self.args.alpha,
                                                                                                        len(self.args.forget_class_idx),
                                                                                                        self.args.paradigm, 
                                                                                                        self.args.cut_sample,
                                                                                                        self.args.file_name), index=False) # 保存结果
        return 




    def FL_round_fusion_selection(num_parties, fusion_key='FedAvg'):
        fusion_class_dict = {
            'FedAvg': FusionAvg(num_parties),
            'Retrain': FusionRetrain(num_parties), 
            'Unlearn': FusionAvg(num_parties)
            }

        return fusion_class_dict[fusion_key]

    