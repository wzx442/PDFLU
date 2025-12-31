import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    # TODO
    parser.add_argument('--model', type=str, required=False, default='LeNet_FashionMNIST', help= 'choose a model: LeNet_FashionMNIST,CNN_Cifar10,CNN_Cifar100')
    parser.add_argument('--data_name', type=str, required=False, default='fashionmnist', help= 'choose: mnist, fashionmnist, cifar10, cifar100')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    parser.add_argument('--distribution', default=True, type=bool, help='True means iid, while False means non-iid')
    parser.add_argument('--train_with_test', default=True, type=bool, help='')
    parser.add_argument('--temperature', default=0.5, type=float, help='the temperature for distillation loss')
    parser.add_argument('--max_checkpoints', default=3, type=int)
    parser.add_argument('--file_name', default='', type=str, help='the name of the experiment')
    # ======================unlearning setting==========================
    # TODO
    parser.add_argument('--forget_paradigm', default='class', type=str, help='choose from client or class or sample')
    parser.add_argument('--paradigm', default='pdflu', type=str, help='choose the training paradigm:pdflu, fused, federaser, retrain, infocom22, exactfun, fl, eraseclient')
    parser.add_argument('--forget_client_idx', type=list, default=[0], help='the index of the client to be forgotten')
    parser.add_argument('--forget_class_idx', type=list, default=[0], help='the index of the class to be forgotten')
    parser.add_argument('--if_retrain', default=False, type=bool, help='')
    parser.add_argument('--if_unlearning', default=False, type=bool, help='')
    parser.add_argument('--baizhanting', default=True, type=bool, help='')
    parser.add_argument('--backdoor', default=False, type=bool, help='')
    parser.add_argument('--backdoor_frac', default=0.2, type=float, help='')
    parser.add_argument('--FU_epoch', default=2, type=int, help='the number of epochs for unlearning')
    # TODO
    parser.add_argument('--MIT', default=True, type=bool, help='whether to use membership inference attack')
    parser.add_argument('--n_shadow', default=5, type=int, help='the number of shadow model')
    parser.add_argument('--cut_sample', default=1.0, type=float, help='using part of the training data')
    parser.add_argument('--relearn', default=False, type=bool, help='whether to relearn the unlearned knowledge')
    parser.add_argument('--save_normal_result', default=True, type=bool, help='whether to save the normal result')
    # ======================batchsize setting===========================
    parser.add_argument('--local_batch_size', default=64, type=int)  # 调整为64以提高训练稳定性
    parser.add_argument('--test_batch_size', default=128, type=int)
    # ======================training epoch===========================
    # TODO
    parser.add_argument('--global_epoch', default=2, type=int)
    parser.add_argument('--local_epoch', default=1, type=int)
    parser.add_argument('--distill_epoch', default=10, type=int)
    parser.add_argument('--distill_pretrain_epoch', default=2, type=int)
    parser.add_argument('--fraction', default=0.8, type=float, help='the fraction of training data')
    parser.add_argument('--num_user', default=10, type=int)
    parser.add_argument('--a', default=0.8, type=float, help='the alpha for lora')
    parser.add_argument('--resnet', default=18, type=int, help='the resnet version')
    # ======================data process============================
    parser.add_argument('--niid', default=True, type=bool, help='')
    parser.add_argument('--balance', default=True, type=bool, help='')
    parser.add_argument('--partition', default='dir', type=str, help='choose from pat or dir')
    parser.add_argument('--alpha', default=1.0, type=float, help='for Dirichlet distribution')
    parser.add_argument('--proxy_frac', default=0.2, type=float, help='the fraction of training data')
    parser.add_argument('--seed', default=50, type=int)
    # parser.add_argument('--only_train', default=False, type=bool)
    # # ======================federaser========================
    parser.add_argument('--unlearn_interval', default=1, type=int, help='')
    parser.add_argument('--forget_local_epoch_ratio', default=0.2, type=float)

    # # ======================eraseclient========================
    parser.add_argument('--epoch_unlearn', default=20, type=int, help='')
    parser.add_argument('--num_iterations', default=50, type=int, help='')



    args = parser.parse_args()
    return args