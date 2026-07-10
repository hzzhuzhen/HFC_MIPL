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




#
## 定义基础命令和路径
#base_command="python main.py --ds FMNIST_MIPL --ds_suffix 1 --epochs 100 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 0"
#base_log_dir="log/FMNIST_1"
#
#learning_rates=(0.03 0.04 0.05 0.06 0.07 0.08)
#
## 遍历学习率数组
#for lr in "${learning_rates[@]}"; do
#    # 格式化学习率为两位小数
#    formatted_lr=$(printf "%.2f" "$lr")
#
#    # 构建命令和日志文件路径
#    command="$base_command --lr $formatted_lr"
#    log_dir="${base_log_dir}/lr_${formatted_lr}"
#    log_file="${log_dir}/FMNIST_MIPL_lrr_0.log"
#
#    # 创建日志目录（如果不存在）
#    mkdir -p "$log_dir"
#
#    # 执行命令并将输出重定向到日志文件
#    eval "$command" > "$log_file"
#
#    # 追加命令行参数到日志文件
#    echo "Command: $command" >> "$log_file"
#done

base_command="python main.py --ds FMNIST_MIPL"
epochs=100
normalize=false
base_log_dir="nlog/FMNIST"

ds_suffixes=(3)
w_entropy_As=(0.0005)
lambda_recon_rates=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
learning_rates=(0.01 0.02 0.03 0.04 0.05)
for ds_suffix in "${ds_suffixes[@]}"; do
    for w_entropy_A in "${w_entropy_As[@]}"; do
        for lambda_recon_rate in "${lambda_recon_rates[@]}"; do
            for lr in "${learning_rates[@]}"; do
                formatted_lr=$(printf "%.2f" "$lr")

                command="$base_command --ds_suffix $ds_suffix --epochs $epochs --normalize $normalize --w_entropy_A $w_entropy_A --lambda_recon_rate $lambda_recon_rate --lr $formatted_lr"
                log_dir="${base_log_dir}_${ds_suffix}/w_entropy_A_${w_entropy_A}/lr_${formatted_lr}"
                log_file="${log_dir}/FMNIST_MIPL_lrr_${lambda_recon_rate}.log"

                mkdir -p "$log_dir"
                eval "$command" > "$log_file"
                echo "Command: $command" >> "$log_file"
            done
        done
    done
done
python main.py --ds SIVAL_MIPL --ds_suffix 1 --normalize false --lr 0.01 --epochs 100 --w_entropy_A 0.001 --lambda_recon_rate 0 --nr_fea 30 --nr_class 25