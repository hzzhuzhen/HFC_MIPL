python main_deepm_v5.py \
    --ds FMNIST_MIPL \
    --ds_suffix 3 \
    --lr 0.01 \
    --epochs 100 \
    --normalize false \
    --w_entropy_A 0.001 \
    --lambda_recon_rate 0.2 \
    --using_lp --using_prep > log/tk_0714_old_pretrain.log


python main_deepm_v5.py \
    --ds FMNIST_MIPL \
    --ds_suffix 3 \
    --lr 0.05 \
    --epochs 100 \
    --nr_fea 784 \
    --nr_class 5 \
    --L 128 \
    --normalize true --w_entropy_A 0.0001 --lambda_recon_rate 0.3 --mu 0.01  --isMixLoss true --mix_loss_rate 0.1 --using_lp --gamma 0.1 --eta 1 --alpha 0.1 --beta 0.00 > log/tk_0714_noLsp.log


python main_deepm_v5.py \
    --ds FMNIST_MIPL \
    --ds_suffix 3 \
    --lr 0.05 \
    --epochs 100 \
    --nr_fea 784 \
    --nr_class 5 \
    --L 128 \
    --normalize true --w_entropy_A 0 --lambda_recon_rate 0.3 --mu 0.01  --isMixLoss true --mix_loss_rate 0.1 --using_lp --gamma 0.1 --eta 1 --alpha 0.1 --beta 0.00 > log/tk_0715_noentropy_a.log