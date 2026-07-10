#!/bin/bash
# | MNIST_MIPL (r=1)    | 0.05          | 0.1   | 1.0  |
# | MNIST_MIPL (r=2)    | 0.05          | 0.1   | 1.0  |
# | MNIST_MIPL (r=3)    | 0.05          | 0.1   | 0.1  |
#python main.py --ds MNIST_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --gamma 0.1 --mu 1. --L 128 > log/MNIST_MIPL_1.log
#python main.py --ds MNIST_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --gamma 0.1 --mu 1. --L 128 > log/MNIST_MIPL_2.log
#python main.py --ds MNIST_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --gamma 0.1 --mu 1. --L 128 > log/MNIST_MIPL_3.log
# | FMNIST_MIPL (r=1)   | 0.01          | 0.5   | 1.0  |
  #| FMNIST_MIPL (r=2)   | 0.01          | 0.5   | 1.0  |

  #| FMNIST_MIPL (r=3)   | 0.05          | 0.5   | 1.0  |
#command="python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2"
#eval "$command" > log/FMNIST_3/lr_1/FMNIST_MIPL_lrr_2.log
#echo "Command: $command" >> log/FMNIST_3/lr_1/FMNIST_MIPL_lrr_2.log
#command="python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.02 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2"
#eval "$command" > log/FMNIST_3/lr_2/FMNIST_MIPL_lrr_2.log
#echo "Command: $command" >> log/FMNIST_3/lr_2/FMNIST_MIPL_lrr_2.log
#command="python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.03 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2"
#eval "$command" > log/FMNIST_3/lr_3/FMNIST_MIPL_lrr_2.log
#echo "Command: $command" >> log/FMNIST_3/lr_3/FMNIST_MIPL_lrr_2.log
#command="python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.04 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2"
#eval "$command" > log/FMNIST_3/lr_4/FMNIST_MIPL_lrr_2.log
#echo "Command: $command" >> log/FMNIST_3/lr_4/FMNIST_MIPL_lrr_2.log
#command="python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2"
#eval "$command" > log/FMNIST_3/lr_5/FMNIST_MIPL_lrr_2.log
#echo "Command: $command" >> log/FMNIST_3/lr_5/FMNIST_MIPL_lrr_2.log
#command="python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.06 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2"
#eval "$command" > log/FMNIST_3/lr_6/FMNIST_MIPL_lrr_2.log
#echo "Command: $command" >> log/FMNIST_3/lr_6/FMNIST_MIPL_lrr_2.log
#command="python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.07 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2"
#eval "$command" > log/FMNIST_3/lr_7/FMNIST_MIPL_lrr_2.log
#echo "Command: $command" >> log/FMNIST_3/lr_7/FMNIST_MIPL_lrr_2.log
#command="python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.08 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2"
#eval "$command" > log/FMNIST_3/lr_8/FMNIST_MIPL_lrr_2.log
#echo "Command: $command" >> log/FMNIST_3/lr_8/FMNIST_MIPL_lrr_2.log
#command="python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.09 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2"
#eval "$command" > log/FMNIST_3/lr_9/FMNIST_MIPL_lrr_2.log
#echo "Command: $command" >> log/FMNIST_3/lr_9/FMNIST_MIPL_lrr_2.log
#command="python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.10 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2"
#eval "$command" > log/FMNIST_3/lr_10/FMNIST_MIPL_lrr_2.log
#echo "Command: $command" >> log/FMNIST_3/lr_10/FMNIST_MIPL_lrr_2.log

#python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 > log/FMNIST_3/lr_1/FMNIST_MIPL_lrr_2.log
#python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.02 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 > log/FMNIST_3/lr_2/FMNIST_MIPL_lrr_2.log
#python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.03 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 > log/FMNIST_3/lr_3/FMNIST_MIPL_lrr_2.log
#python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.04 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 > log/FMNIST_3/lr_4/FMNIST_MIPL_lrr_2.log
#python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 > log/FMNIST_3/lr_5/FMNIST_MIPL_lrr_2.log
#python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.06 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 > log/FMNIST_3/lr_6/FMNIST_MIPL_lrr_2.log
#python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.07 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 > log/FMNIST_3/lr_7/FMNIST_MIPL_lrr_2.log
#python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.08 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 > log/FMNIST_3/lr_8/FMNIST_MIPL_lrr_2.log
#python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.09 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 > log/FMNIST_3/lr_9/FMNIST_MIPL_lrr_2.log
#python main.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.10 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 > log/FMNIST_3/lr_10/FMNIST_MIPL_lrr_2.log

# | Birdsong_MIPL (r=1) | 0.01          | 10.   | 10.  |
  #| Birdsong_MIPL (r=2) | 0.01          | 10.   | 10.  |
  #| Birdsong_MIPL (r=3) | 0.01          | 10.   | 10.  |
#python main.py --ds Birdsong_MIPL --ds_suffix 1 --normalize false --lr 0.01 --epochs 100 --gamma 10. --mu 1. --L 128 > log/Birdsong_MIPL_1.log
#python main.py --ds Birdsong_MIPL --ds_suffix 2 --normalize false --lr 0.01 --epochs 100 --gamma 10. --mu 1. --L 128 > log/Birdsong_MIPL_2.log
#python main.py --ds Birdsong_MIPL --ds_suffix 3 --normalize false --lr 0.01 --epochs 100 --gamma 10. --mu 1. --L 128 > log/Birdsong_MIPL_3.log
# | SIVAL_MIPL (r=1)    | 0.01          | 10.   | 10.  |
  #| SIVAL_MIPL (r=2)    | 0.01          | 10.   | 10.  |
  #| SIVAL_MIPL (r=3)    | 0.05          | 10.   | 10.  |
#python main.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --gamma 10. --mu 10. --L 128 > log/SIVAL_MIPL_1.log
#python main.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --gamma 10. --mu 10. --L 128 > log/SIVAL_MIPL_2.log
#python main.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --gamma 10. --mu 10. --L 128 > log/SIVAL_MIPL_3.log
## Parameter Settings
#| Dataset             | learning rate | gamma | mu   |
#| ------------------- | ------------- | ----- | ---- |
#| CRC-MIPL-Row        | 0.01          | 10.   | 10.  |
#| CRC-MIPL-SBN        | 0.01          | 10.   | 10.  |
#| CRC-MIPL-KMeansSeg  | 0.01          | 10.   | 10.  |
#| CRC-MIPL-SIFT       | 0.01          | 10.   | 10.  |
#python main.py --ds CRC-MIPL-Row --normalize false --lr 0.05 --epochs 100 --gamma 10. --mu 10. --L 128 > log/CRC-MIPL-Row.log
#python main.py --ds CRC-MIPL-SBN --normalize false --lr 0.05 --epochs 100 --gamma 10. --mu 10. --L 128 > log/CRC-MIPL-SBN.log
#python main.py --ds CRC-MIPL-KMeansSeg --normalize false --lr 0.05 --epochs 100 --gamma 10. --mu 10. --L 128 > log/CRC-MIPL-KMeansSeg.log
#python main.py --ds CRC-MIPL-SIFT --normalize false --lr 0.05 --epochs 100 --gamma 10. --mu 10. --L 128 > log/CRC-MIPL-SIFT.log

#python main_deepm_v3.py --ds MNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.9 --mu 0.2  --mu2 0.001 --la 2 > ../../log/MNIST_MIPL_1_deepm_sp.log
#python main_deepm_v3.py --ds MNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.9 --mu 0.2  --mu2 0.001 --la 2 > ../../log/MNIST_MIPL_2_deepm_sp.log
#python main_deepm_v3.py --ds MNIST_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.9 --mu 0.2  --mu2 0.001 --la 2 > ../../log/MNIST_MIPL_3_deepm_sp.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.85 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.85 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.85 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp.log

# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --nr_fea 30 --nr_class 25 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.85 --mu 10.0  --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 30 --nr_class 25 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.85 --mu 10.0  --mu2 0.001 --la 2 > ../../log/SIVAL_MIPLL_2_deepm_sp.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.85 --mu 10.0  --mu2 0.001 --la 2 > ../../og/SIVAL_MIPL_3_deepm_sp.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.85 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu10.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.85 --mu 0.2  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.85 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu10.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.85 --mu 0.2  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02.log

#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.9 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02_recon09.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02_recon05.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 1.5 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02_recon15.log

#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.1 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02_recon01.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02_recon05.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.5 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02_recon07.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.08 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.5 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr008_mu02_recon07.log

#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02_recon02.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02_recon03.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.4 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02_recon04.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.5 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_mu02_recon05.log

#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.15 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon015.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.25 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon025.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.002 --lambda_recon_rate 0.2 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a002.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0000 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu00_recon00_a0000（baseline）.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 0.5  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu05.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.5  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu15.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 2.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu20.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 4.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu40.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 6.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu60.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 10.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu100.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu10.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu11.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.2  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu12.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu13.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.4  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu14.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.5  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu15.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.6  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu16.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.7  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu17.log #最佳
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.8  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu18.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.9  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu19.log

# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.7  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon02_a0005_mu17.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon02_a0005_mu10.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon02_a0005_mu03.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.9 --mu 1.7  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon09_a0005_mu17.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.9 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon09_a0005_mu10.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.9 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon09_a0005_mu03.log

# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon02_a0005_mu03.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr001_mu02_recon02_a0005_mu03.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.1 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon01_a0005_mu03.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.3 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon03_a0005_mu03.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 0.2  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon02_a0005_mu02.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 0.4  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon02_a0005_mu04.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon02_a001_mu03.log


#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu02_recon02_a0005_mu03.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.5 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu02_recon05_a0005_mu03.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu02_recon02_a0005_mu10.log

# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.1 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu01_recon02_a0005_mu10.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.3 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu03_recon02_a0005_mu10.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.5 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu05_recon02_a0005_mu10.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 1.0 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu10_recon02_a0005_mu10.log

#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.1 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu01_recon02_a0005_mu03.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.3 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu03_recon02_a0005_mu03.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.5 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu05_recon02_a0005_mu03.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 1.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_mu10_recon02_a0005_mu03.log

#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 2.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon20_a0005_mu03.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 5.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon50_a0005_mu03.log

# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 10.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon100_a0005_mu03-compare1.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon100_a001_mu03-compare2.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 20.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon200_a0005_mu03.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 50.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon500_a0005_mu03.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 100.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon1000_a0005_mu03.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 200.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon2000_a0005_mu03.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 300.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon3000_a0005_mu03.log

# 调 流行参数
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon100_a001_mu03.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 8.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon80_a001_mu03.log #
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu03.log # 最好
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 12.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon120_a001_mu03.log


#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 7.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon70_a001_mu03.log #
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu03.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.005 --lambda_recon_rate 6.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a005_mu03.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.010 --lambda_recon_rate 6.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a01_mu03.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.05 --lambda_recon_rate 6.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a05_mu03.log 
# 调注意力#
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0007 --lambda_recon_rate 6.0 --mu 0.3  --mu2 0.0009 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a0007_mu03.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0009 --lambda_recon_rate 6.0 --mu 0.3  --mu2 0.0009 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a0009_mu03.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0011 --lambda_recon_rate 6.0 --mu 0.3  --mu2 0.0009 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a0011_mu03.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.0015 --lambda_recon_rate 6.0 --mu 0.3  --mu2 0.0009 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a0015_mu03.log 

# 调稀疏参数
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01.log #
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu03.log #
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 1.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu10.log #
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 2.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu20.log #
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.01  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu001.log 
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.001  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu0001.log 
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.00  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu000.log 
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.15  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu015.log 
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.12  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu012.log 
# 学习率
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01.log 
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.04 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr004_recon60_a001_mu01.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.06 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr006_recon60_a001_mu01.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.08 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr008_recon60_a001_mu01.log 
# # 调中间维度大小
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01_L128.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 256 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01_L256.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 512 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01_L512.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 1024 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01_L1024.log 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 2048 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01_L2048.log 


#SIVAL_MIPL
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01.log 

#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon60_a001_mu10_e200.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon60_a001_mu10_e200.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon60_a001_mu10_e200.log

# 流行学习
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon02_a001_mu10.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.9 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon09_a001_mu10.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 1.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon10_a001_mu10.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 3.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon30_a001_mu10.log

#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 5.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon50_a001_mu10.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu10.log
# 学习率检测（0.1 0.01  0.0005）
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.005 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr0005_recon100_a001_mu10.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.01 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr001_recon100_a001_mu10.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.1 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr01_recon100_a001_mu10.log

# 稀疏参数
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.0 --mu 0.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon00_a001_mu0（baseline）.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.1 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu01.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu03.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.5 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu05.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.5 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu15.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 3.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu30.log

# 参数层参数
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu03.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 256 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu03_256.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 512 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu03_512_e200.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 1024 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu03_1024_e200.log

#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 2048 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu03_2048_e200.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 4096 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr005_recon100_a001_mu03_4096.log


# SIVAL2
#SIVAL_MIPL

# # 流行学习
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon02_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.9 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon09_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 1.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon10_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 3.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon30_a001_mu10.log

# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 5.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon50_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu10.log
# # 学习率检测（0.1 0.01  0.0005）
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.005 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr0005_recon100_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.01 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr001_recon100_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.1 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr01_recon100_a001_mu10.log

# # 稀疏参数
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.0 --mu 0.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon00_a001_mu0（baseline）.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.1 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu01.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu03.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.5 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu05.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.5 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu15.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 3.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu30.log

# # 参数层参数
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu03.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 256 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu03_256.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 512 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu03_512_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 1024 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu03_1024.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 2048 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu03_2048_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 4096 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu03_4096.log

# # SIVAL3
# #SIVAL_MIPL

# # 流行学习
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon02_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.9 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon09_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 1.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon10_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 3.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon30_a001_mu10.log

# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 5.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon50_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu10.log
# # 学习率检测（0.1 0.01  0.0005）
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.005 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr0005_recon100_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.01 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr001_recon100_a001_mu10.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.1 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr01_recon100_a001_mu10.log

# # 稀疏参数
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.0 --mu 0.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon00_a001_mu0（baseline）.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.1 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu01.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu03.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.5 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu05.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 1.5 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu15.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 3.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu30.log

# # 参数层参数
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu03.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 256 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu03_256_e200.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 512 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu03_512_e200.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 1024 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu03_1024_e200.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 2048 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu03_2048_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 100 --nr_fea 30 --nr_class 25 --L 4096 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon100_a001_mu03_4096.log


#补充测试 L100效果上来了
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 100 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.7  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu17_L100.log 
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 100 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.7  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr005_mu02_recon02_a0005_mu17_L100.log 
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 100 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon02_a0005_mu03_L100.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01_L100.log 
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 256 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01_L256.log 
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 512 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 0.1  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_lr005_recon60_a001_mu01_L512.log 

# python main_deepm_v3.py --ds MNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --L 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.9 --mu 0.2  --mu2 0.001 --la 2 > ../../log/MNIST_MIPL_1_rate_09_100.log  #结果超神
#python main_deepm_v3.py --ds MNIST_MIPL --ds_suffix 2 --epochs 100  --lr 0.05 --normalize false  --w_entropy_A 0.001 --nr_fea 784 --nr_class 5 --L 300 --lambda_recon_rate 0.85 --mu 1.0  --mu2 0.001 --la 2 > ../../log/MNIST_MIPL_2_recon085_mu1_300.log
#python main_deepm_v3.py --ds MNIST_MIPL --ds_suffix 2 --epochs 100  --lr 0.05 --normalize false  --w_entropy_A 0.001 --nr_fea 784 --nr_class 5 --L 100 --lambda_recon_rate 0.85 --mu 1.0  --mu2 0.001 --la 2 > ../../log/MNIST_MIPL_2_recon085_mu1_100.log
#python main_deepm_v3.py --ds MNIST_MIPL --ds_suffix 2 --epochs 100  --lr 0.05 --normalize false  --w_entropy_A 0.001 --nr_fea 784 --nr_class 5 --L 200 --lambda_recon_rate 0.85 --mu 1.0  --mu2 0.001 --la 2 > ../../log/MNIST_MIPL_2_recon085_mu1_200.log
#python main_deepm_v3.py --ds MNIST_MIPL --ds_suffix 2 --epochs 100  --lr 0.05 --normalize false  --w_entropy_A 0.001 --nr_fea 784 --nr_class 5 --L 400 --lambda_recon_rate 0.85 --mu 1.0  --mu2 0.001 --la 2 > ../../log/MNIST_MIPL_2_recon085_mu1_400.log

#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon02_a001_mu03_L100.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr005_mu02_recon02_a001_mu03_L128.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 100 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_lr001_recon02_a0005_mu03_L100.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 100 --normalize false --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 1.7  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_1_deepm_sp_lr001_recon02_a0005_mu17_L100.log

#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon02_a0001_mu03_L500.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize true --w_entropy_A 0.001 --L 500 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_baseline.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_baseline.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_baseline.log

#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --lr 0.01 --epochs 200 --normalize false --w_entropy_A 0.001 --L 500 --nr_fea 30 --nr_class 25 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_baseline.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 2 --lr 0.05 --epochs 200 --normalize false --w_entropy_A 0.001 --L 500 --nr_fea 30 --nr_class 25 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_2_baseline.log
#python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --lr 0.05 --epochs 200 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 30 --nr_class 25 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_baseline.log

# CRC太慢了 夜里再跑
##python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_baseline.log
#python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_baseline.log
#python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_baseline.log
#python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_baseline.log

#根据新参数重跑 FMNIST_MIPL r2
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 100 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr005_mu02_recon02_a001_mu03_L100.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 100 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_mu02_recon02_a001_mu03_L100.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_mu02_recon02_a001_mu03_L500.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon00_a001_mu00_L128.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.0005 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon00_a0005_mu00_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon00_a001_mu00_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.1 --mu 0.0  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon01_a001_mu00_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 0.0  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon02_a001_mu00_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 0.0  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon03_a001_mu00_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.5 --mu 0.0  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon05_a001_mu00_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.9 --mu 0.0  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon09_a001_mu00_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 1.2 --mu 0.0  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon12_a001_mu00_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 1.7 --mu 0.0  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon17_a001_mu00_L128.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 0.0  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon02_a001_mu00_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 0.1  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon03_a001_mu01_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 0.2  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon03_a001_mu02_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 0.3  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon03_a001_mu03_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 0.5  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon03_a001_mu05_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 0.9  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon03_a001_mu09_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 1.2  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon03_a001_mu12_L128.log
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.3 --mu 1.7  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_2_deepm_sp_normalize_lr001_recon03_a001_mu17_L128.log
#根据新参数重跑 FMNIST_MIPL r3
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize true --w_entropy_A 0.0001 --lambda_recon_rate 0.0 --mu 0.00  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_normalize_baseline.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 128 --normalize true --w_entropy_A 0.0001 --lambda_recon_rate 0.3 --mu 0.01  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_normalize_lr001_recon03_a001_mu001_L128.log
#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 3 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 500 --normalize true --w_entropy_A 0.0001 --lambda_recon_rate 0.3 --mu 0.01  --mu2 0.000 --la 2 > ../../log/FMNIST_MIPL_3_deepm_sp_normalize_lr001_recon03_a001_mu001_L500.log

# CRC太慢了 夜里再跑
#python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_baseline.log
#python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_baseline.log
#python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_baseline.log
#python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_baseline.log

# 先跑birdsong，后修复数据集问题
# birdsong 1
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_baseline.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon02_a0001_mu03_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 100 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon02_a0001_mu03_L100.log

#python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize true --w_entropy_A 0.001 --L 500 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/FMNIST_MIPL_2_baseline.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_baseline.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_baseline.log

#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.01 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon001_a0001_mu00_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon01_a0001_mu00_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.2 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon02_a0001_mu00_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon03_a0001_mu00_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.4 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon04_a0001_mu00_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.5 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon05_a0001_mu00_L500.log

#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.1  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon03_a0001_mu01_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.2  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon03_a0001_mu02_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.3  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon03_a0001_mu03_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon03_a0001_mu05_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.7  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon03_a0001_mu07_L500.log
#python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.9  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_1_lr001_recon03_a0001_mu09_L500.log



# birdsong 2,3
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon00_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon01_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.2 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon02_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.4 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon04_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.5 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon05_a0001_mu00_L500.log

# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon00_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon01_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.2 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon02_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon03_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.4 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon04_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.5 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon05_a0001_mu00_L500.log

# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.1  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu01_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.2  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu02_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.3  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu03_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.4  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu04_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu05_L500.log

# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.1  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu01_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.2  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu02_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.3  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu03_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.4  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu04_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu05_L500.log

# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.6 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon06_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.7 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon07_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.8 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon08_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.9 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon09_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon10_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.1 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon11_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.2 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon12_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.3 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon13_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.4 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon14_a0001_mu00_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.5 --mu 0.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon15_a0001_mu00_L500.log



# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.01  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu001_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.9  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu09_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 1.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu15_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.5 --mu 0.2  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon05_a0001_mu02_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.5 --mu 0.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon05_a0001_mu05_L500.log


# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 0.1  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon01_a0001_mu01_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 0.2  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon01_a0001_mu02_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 0.3  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon01_a0001_mu03_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 0.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon01_a0001_mu05_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 0.9  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon01_a0001_mu09_L500.log

# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.6 --mu 0.1  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon06_a0001_mu01_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.6 --mu 0.2  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon06_a0001_mu02_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.6 --mu 0.3  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon06_a0001_mu03_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.6 --mu 0.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon06_a0001_mu05_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.6 --mu 0.9  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon06_a0001_mu09_L500.log

# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.1 --mu 0.1  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon11_a0001_mu01_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.1 --mu 0.2  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon11_a0001_mu02_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.1 --mu 0.3  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon11_a0001_mu03_L500.log



# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 1.4  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu14_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 1.6  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu16_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 1.7  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu17_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 1.9  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu19_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 2.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu20_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 2.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu25_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 3.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu30_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 4.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu40_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 5.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_2_lr001_recon03_a0001_mu50_L500.log


# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 1.6  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon01_a0001_mu16_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 1.9  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon01_a0001_mu19_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 2.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon01_a0001_mu20_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 2.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon01_a0001_mu25_L500.log

# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.6 --mu 1.6  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon06_a0001_mu16_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.6 --mu 1.9  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon06_a0001_mu19_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.6 --mu 2.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon06_a0001_mu20_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.6 --mu 2.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon06_a0001_mu25_L500.log

# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.1 --mu 1.6  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon11_a0001_mu16_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.1 --mu 1.9  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon11_a0001_mu19_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.1 --mu 2.0  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon11_a0001_mu20_L500.log
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 1.1 --mu 2.5  --mu2 0.001 --la 2 > ../../log/Birdsong_MIPL_3_lr001_recon11_a0001_mu25_L500.log

# CRC太慢了 夜里再跑
#python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_baseline.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.1 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon01_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.2 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon02_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.3 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon03_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.5 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon05_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.9 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon09_a0001_mu00_L500.log


#python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_baseline.log

# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.3 --mu 1.5  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon03_a0001_mu15_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.5 --mu 1.5  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon05_a0001_mu15_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.5 --mu 2.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon05_a0001_mu20_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.5 --mu 2.5  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon05_a0001_mu25_L500.log

# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.5 --mu 1.6  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon05_a0001_mu16_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.5 --mu 1.4  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon05_a0001_mu14_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.5 --mu 1.7  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon05_a0001_mu17_L500.log


# 参数层参数
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.01 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr001_recon60_a001_mu10_128_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.01 --epochs 200 --nr_fea 30 --nr_class 25 --L 256 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr001_recon60_a001_mu10_256_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.01 --epochs 200 --nr_fea 30 --nr_class 25 --L 512 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr001_recon60_a001_mu10_512_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.01 --epochs 200 --nr_fea 30 --nr_class 25 --L 2048 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 6.0 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_1_deepm_sp_lr001_recon60_a001_mu10_2048_e200.log


# SIVAL_MIPL 3
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.2 --mu 0.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon02_a0001_mu00_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.5 --mu 0.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon05_a0001_mu00_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.7 --mu 0.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon07_a0001_mu00_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.9 --mu 0.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon09_a0001_mu00_e200.log

# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.7 --mu 0.1 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon07_a0001_mu01_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.7 --mu 0.3 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon07_a0001_mu03_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.7 --mu 0.5 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon07_a0001_mu05_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.7 --mu 1.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon07_a0001_mu10_e200.log


# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.7 --mu 0.7 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon07_a0001_mu07_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.7 --mu 1.5 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon07_a0001_mu15_e200.log
# python main_deepm_v3.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.7 --mu 2.0 --mu2 0.001 --la 2 > ../../log/SIVAL_MIPL_3_deepm_sp_lr005_recon07_a0001_mu20_e200.log


# CRC太慢了 夜里再跑
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_baseline.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.1 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon01_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.2 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon02_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.3 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon03_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.5 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon05_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.9 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-Row_8_8_9_lr001_recon09_a0001_mu00_L500.log


# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.2 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon02_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.3 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon03_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.5 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon05_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.7 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon07_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.9 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon09_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.15 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon15_a0001_mu00_L500.log

# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.7 --mu 0.2  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon07_a0001_mu02_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.7 --mu 0.5  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon07_a0001_mu05_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.7 --mu 0.7  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon07_a0001_mu07_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.7 --mu 1.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon07_a0001_mu10_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.7 --mu 1.5  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon07_a0001_mu15_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.7 --mu 2.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon07_a0001_mu20_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.7 --mu 10.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SBN_8_9_15_lr001_recon07_a0001_mu100_L500.log

#python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_baseline.log

# python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.2 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_lr001_recon02_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.4 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_lr001_recon04_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.6 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_lr001_recon06_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.7 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_lr001_recon07_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.9 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_lr001_recon09_a0001_mu00_L500.log

# python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.7 --mu 0.1  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_lr001_recon07_a0001_mu01_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.7 --mu 0.2  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_lr001_recon07_a0001_mu02_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.7 --mu 0.5  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_lr001_recon07_a0001_mu05_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.7 --mu 1.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_lr001_recon07_a0001_mu10_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-KMeansSeg --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 6 --nr_class 7 --lambda_recon_rate 0.7 --mu 1.5  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-KMeansSeg_n_6_lr001_recon07_a0001_mu15_L500.log



# 回归测试
# python main_deepm_v3.py --ds MNIST_MIPL --ds_suffix 1 --normalize false --lr 0.05 --epochs 100 --gamma 0.1 --mu 1. --L 128 
# python main_deepm_v3.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.1  --mu2 0.001 --la 2 
# python main_deepm_v3.py --ds FMNIST_MIPL --ds_suffix 2 --lr 0.05 --epochs 100 --nr_fea 784 --nr_class 5 --L 100 --normalize true --w_entropy_A 0.001 --lambda_recon_rate 0.2 --mu 0.3  --mu2 0.001 --la 2


# 最后个数据集
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.0 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_baseline.log
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.2 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon02_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.3 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon03_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.5 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon05_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.7 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon07_a0001_mu00_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.9 --mu 0.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon09_a0001_mu00_L500.log


# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.5 --mu 0.3  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon05_a0001_mu03_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.5 --mu 0.5  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon05_a0001_mu05_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.5 --mu 1.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon05_a0001_mu10_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.5 --mu 1.5  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon05_a0001_mu15_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.5 --mu 2.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon05_a0001_mu20_L500.log
# python main_deepm_v3.py --ds CRC-MIPL-SIFT --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 128 --nr_class 7 --lambda_recon_rate 0.5 --mu 3.0  --mu2 0.001 --la 2 > ../../log/CRC-MIPL-SIFT_25_128_lr001_recon05_a0001_mu30_L500.log


python main_deepm_v3.py --ds MNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0.9 --mu 0.2  --mu2 0.001 --la 2 > ../../log/MNIST_MIPL_1_mix_loss_deepm_sp.log