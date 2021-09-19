cd ..

python train.py \
--exp_name 15.privileged \
--model weakly \
--batch_size 8 \
--w_bde 3000 \
--w_l2n 3000 \
--w_gmi 0.5 \
--gpu 0 \
--mv_mod mixed \
--fx_mod t2 \
--mi_resample 1 \
--mi_resample_count 5
