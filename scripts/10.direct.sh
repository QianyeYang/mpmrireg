cd ..

python train.py \
--exp_name 10.direct \
--model origin \
--batch_size 8 \
--num_epochs 4000 \
--w_bde 3000 \
--w_l2n 3000 \
--w_gmi 0.5 \
--gpu 0 \
--mv_mod dwi
