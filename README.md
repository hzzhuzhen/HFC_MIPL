



## Requirements

```sh
numpy==1.21.5
scikit_learn==1.3.1
scipy==1.7.3
torch==1.12.1+cu113
```

To install the requirement packages, please run the following command:

```sh
pip install -r requirements.txt
```



## Datasets

The datasets used in this paper can be found on this [link](http://palm.seu.edu.cn/zhangml/Resources.htm#MIPL_data).



## Demo

To reproduce the results of MNIST_MIPL dataset in the paper, please run the following command:

```sh
python main_hfc.py --ds MNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.001
```



This package is only free for academic usage. Have fun!



## Parameter Settings

| Dataset             | learning rate | 
| ------------------- | ------------- | 
| MNIST_MIPL (r=1)    | 0.05          | 
| MNIST_MIPL (r=2)    | 0.05          | 
| MNIST_MIPL (r=3)    | 0.05          | 
| FMNIST_MIPL (r=1)   | 0.01          | 
| FMNIST_MIPL (r=2)   | 0.01          | 
| FMNIST_MIPL (r=3)   | 0.05          | 
| Birdsong_MIPL (r=1) | 0.01          | 
| Birdsong_MIPL (r=2) | 0.01          | 
| Birdsong_MIPL (r=3) | 0.01          | 
| SIVAL_MIPL (r=1)    | 0.01          | 
| SIVAL_MIPL (r=2)    | 0.01          | 
| SIVAL_MIPL (r=3)    | 0.05          | 
| CRC-MIPL-Row        | 0.01          | 
| CRC-MIPL-SBN        | 0.01          | 
| CRC-MIPL-KMeansSeg  | 0.01          | 
| CRC-MIPL-SIFT       | 0.01          | 



This package is only free for academic usage. Have fun!


MNIST_MIPL
(500,4)   1300个样本，4维度，分别：特征数据，偏标签，真实标签；
(bag_num,784) 包内数据格式,特征维度784维，标签类别5个

FMNIST_MIPL
(500,4)   1300个样本，4维度，分别：特征数据，偏标签，真实标签；
(bag_num,784) 包内数据格式， 特征维度784维，标签类别5个

bridsong
(1300,4)   1300个样本，4维度，分别：特征数据，偏标签，真实标签；
(bag_num,38) 包内数据格式， 特征维度38维，标签类别13个

SIVAL_MIPL
(1500,4)   1300个样本，4维度，分别：特征数据，偏标签，真实标签；
(bag_num,30) 包内数据格式， 特征维度30维，标签类别25个

CRC-MIPL-Row
(7000,3)   1300个样本，4维度，分别：特征数据，偏标签，真实标签；
(bag_num,9) 包内数据格式， 特征维度9维，标签类别7个