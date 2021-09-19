cd ..

python train.py \
--exp_name 14.mixed \
--model origin \
--batch_size 8 \
--w_bde 3000 \
--w_l2n 3000 \
--w_gmi 0.5 \
--gpu 2 \
--mv_mod mixed \
--fx_mod t2
