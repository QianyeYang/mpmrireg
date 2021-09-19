cd ..

python train.py \
--exp_name 18.joint \
--model joint3 \
--batch_size 8 \
--w_bde 3000 \
--w_l2n 3000 \
--w_gmi 0.5 \
--w_dsl 100 \
--gpu 0
