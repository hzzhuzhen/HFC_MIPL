# python main_hfc.py --ds MNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --isMixLoss true --w_entropy_A 0.001 --lambda_recon_rate 0.84 --mu 1.0 --mix_loss_rate 0.1  --L 100  --isMixLoss true --using_lp --gamma 0.001 --eta 0.001 --alpha 0.001 --beta 0.00 > log/V_nolabelgraph_MNIST_MIPL_1_mix_loss_deepm_sp_normalize_lr001_recon84_recon002_a001_mu100_L100_mix010_lp_g001_e001_a001_b00.log 
python main_hfc.py --ds MNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --isMixLoss true --w_entropy_A 0.001 --lambda_recon_rate 0.84 --mu 0.001 --mix_loss_rate 0.01  --L 100  --isMixLoss true --using_lp> log/V_nolabelgraph_MNIST_MIPL_1_mix_loss_deepm_sp_normalize_lr001_recon84_recon002_a001_mu100_L100_mix010_lp_g001_e001_a001_b00.log 

python main_hfc.py --ds MNIST_MIPL --ds_suffix 2 --epochs 100  --lr 0.05 --normalize false  --w_entropy_A 0.001 --nr_fea 784 --nr_class 5 --L 133 --mu 1.0 --lambda_recon_rate 0.84 --mix_loss_rate 0.1 --isMixLoss true --using_lp --gamma 0.1 --eta 0.1 --alpha 0.001 --beta 0.00 >  log/V_nolabelgraph_MNIST_MIPL_2_mix_loss_deepm_sp_normalize_lr001_recon84_recon020_a001_mu100_L133_mix010_lp_g01_e01_a001_b00 
#python main_hfc.py --ds MNIST_MIPL --ds_suffix 3 --epochs 100  --lr 0.05 --normalize false  --w_entropy_A 0.001 --nr_fea 784 --nr_class 5 --L 99 --mu 0.2 --lambda_recon_rate 0.2 --mix_loss_rate 0.001 --isMixLoss true --using_lp --gamma 0.1 --eta 0.1 --alpha 0.1 --beta 0.00 >  log/V_nolabelgraph_MNIST_MIPL_3_mix_loss_deepm_sp_normalize_lr001_recon20_2recon001_a001_mu020_L99_mix001_lp_g01_e01_a01_b00

#python -m debugpy --listen localhost:5678 --wait-for-client main_hfc.py --ds MNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --isMixLoss true --w_entropy_A 0.001 --lambda_recon_rate 0.84 --mu 0.001 --mix_loss_rate 0.1  --L 100  --isMixLoss true --using_lp
















































# python main_deepm_v5.py --ds MNIST_MIPL --ds_suffix 3 --epochs 100  --lr 0.05 --normalize false  --w_entropy_A 0.001 --nr_fea 784 --nr_class 5 --L 100 --lambda_recon_rate 0.2 --mu 0.2  --isMixLoss true --mix_loss_rate 0.001 >  ../../log/A_MNIST_MIPL_3_recon02_mu02_L100_mix0001.log 
#python main_deepm_v5.py --ds MNIST_MIPL --ds_suffix 3 --epochs 100  --lr 0.05 --normalize false  --w_entropy_A 0.001 --nr_fea 784 --nr_class 5 --L 100 --lambda_recon_rate 0.2 --mu 0.2  --isMixLoss true --mix_loss_rate 0.001 --lambda_recon_rate 0.83 --mu 1.0 --second_lambda_recon_rate 0.02 --mu 1.8  --isMixLoss true --using_lp --gamma 0.1 --eta 1 --alpha 0.1 --beta 0.00 >  ../../log/A_MNIST_MIPL_3_recon02_mu02_L100_mix0001.log 

# python main_deepm_v5.py --ds Birdsong_MIPL --ds_suffix 1 --lr 0.01 --epochs 200 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 0.3 --isMixLoss true --mix_loss_rate 0.1 > ../../log/A_Birdsong_MIPL_1_lr001_recon03_a0001_mu03_L500_mix01.log
# python main_deepm_v5.py --ds Birdsong_MIPL --ds_suffix 2 --lr 0.01 --epochs 200 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.3 --mu 2.0 --isMixLoss true --mix_loss_rate 0.1 > ../../log/A_Birdsong_MIPL_2_lr001_recon03_a0001_mu20_L500_mix01.log
# python main_deepm_v5.py --ds Birdsong_MIPL --ds_suffix 3 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.0001 --L 500 --nr_fea 38 --nr_class 13 --lambda_recon_rate 0.1 --mu 1.6 --isMixLoss true --mix_loss_rate 0.1 > ../../log/A_Birdsong_MIPL_3_lr001_recon01_a0001_mu16_L500_mix01.log
# python main_deepm_v5.py --ds SIVAL_MIPL --ds_suffix 2 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.001 --lambda_recon_rate 10.0 --mu 0.3 --isMixLoss true --mix_loss_rate 0.1 > ../../log/A_SIVAL_MIPL_2_deepm_sp_lr005_recon100_a001_mu03_128_e200_mix01.log
# python main_deepm_v5.py --ds SIVAL_MIPL --ds_suffix 3 --normalize false --lr 0.05 --epochs 200 --nr_fea 30 --nr_class 25 --L 128 --normalize false --w_entropy_A 0.0001 --lambda_recon_rate 0.7 --mu 2.0 --isMixLoss true --mix_loss_rate 0.1 > ../../log/A_SIVAL_MIPL_3_deepm_sp_lr005_recon07_a0001_mu20_e200_mix01.log
# python main_deepm_v5.py --ds CRC-MIPL-Row --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.001 --L 500 --nr_fea 9 --nr_class 7 --lambda_recon_rate 0.3 --mu 1.5  --isMixLoss true --mix_loss_rate 0.5 > ../../log/A_CRC-MIPL-Row_8_8_9_lr001_recon03_a0001_mu15_L500_mix05.log #best
# python main_deepm_v5.py --ds CRC-MIPL-SBN --lr 0.01 --epochs 200 --normalize true --w_entropy_A 0.0001 --L 500 --nr_fea 15 --nr_class 7 --lambda_recon_rate 0.7 --mu 1.5  --isMixLoss true --mix_loss_rate 0.5 > ../../log/A_CRC-MIPL-SBN_8_9_15_lr001_recon07_a0001_mu15_L500_mix05.log #better


python main.py --ds MNIST_MIPL --ds_suffix 1 --lr 0.01 --epochs 100 --normalize false --w_entropy_A 0.001 >  ../../log/MINST_baseline
#--epochs 100  --lr 0.05 --normalize false  --w_entropy_A 0.001 --nr_fea 784 --nr_class 5 --L 99 --mu 0.2 --lambda_recon_rate 0.2 --mix_loss_rate 0.001 --second_lambda_recon_rate 0.01 --isMixLoss true --using_lp --gamma 0.1 --eta 0.1 --alpha 0.1 --beta 0.00 >  ../../log/V_nolabelgraph_MNIST_MIPL_3_mix_loss_deepm_sp_normalize_lr001_recon20_2recon001_a001_mu020_L99_mix001_lp_g01_e01_a01_b00 best