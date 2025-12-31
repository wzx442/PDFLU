# Privacy-Preserving Federated Learning and Unlearning Supporting Dynamic User
<!-- Welcome to the repository! This is the detailed implementation of our project, FUSED. We hope this code will serve as a valuable resource for understanding our work and its application. Thank you for your interest and support! -->
<!-- ![img_1.png](img_1.png) -->
## Dependencies
```
torch==2.2.1+cu121
numpy==1.24.3
scikit-learn==1.3.2
objgraph==3.6.1
pandas==2.0.2
torchvision==0.17.0+cu121
joblib==1.3.2
transformers==4.37.2
```
## Datasets
### Image Datasets
-[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)  
-[Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)  
-[FashionMNIST](https://www.worldlink.com.cn/en/osdir/fashion-mnist.html)

## Quick start
```bash
python main.py --data_name='fashionmnist' --forget_paradigm='client' --paradigm='pdflu'  --global_epoch=100 --FU_epoch=100 --local_epoch=5 --fraction=1 --alpha=1.0 --num_user=100 --a=0.8 --file_name=fashionmnist-50user-100epoch-pdflu-client
```

```bash
python main.py --data_name='cifar10' --forget_paradigm='client' --paradigm='pdflu'  --global_epoch=100 --FU_epoch=1 --local_epoch=5 --alpha=1.0 --fraction=1 --num_user=5 --a=0.8 --file_name=cifar10-5user-noniid-test
```

- FedAvg 
```bash
python main.py --data_name='cifar10' --forget_paradigm='client' --paradigm='fedavg' --only_train=True --global_epoch=100 --FU_epoch=1 --local_epoch=5 --alpha=1.0 --fraction=1 --num_user=100 --a=0.8 --file_name=cifar10-100user-noniid
```

<!-- ## results文件夹 -->
<!-- 根据不同的遗忘范式(client, class, sample), 会在不同的子文件夹内存放结果csv -->

## description of forget_paradigm

if --forget_paradigm=client, run client unlearning

--forget_client_idx=[], forgetting the data of clients with the specified indices

--------

if --forget_paradigm=class, class unlearning

--forget_class_idx=[], forgett forgetting the label classes of **all clients** with the specified indices

<!-- 样本遗忘存在nan的情况, 是因为数据量分配有时有问题, 重新运行即可 -->
