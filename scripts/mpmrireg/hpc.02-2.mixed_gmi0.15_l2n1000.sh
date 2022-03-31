#$ -S /bin/bash
#$ -l tmem=12G
#$ -l gpu=true,gpu_type=titanxp
#$ -l h_rt=300:0:0
#$ -j y
#$ -N mix_m.15_g1e3
#$ -cwd
hostname
date
python3 -u train.py \
--project mpmrireg \
--exp_name 02-2.mixed_gmi0.15_l2n1000 \
--data_path ../data/mpMriReg/FinalProcessed-v2/52-52-46-ldmk \
--method mixed \
--w_l2g 1000 \
--w_gmi 0.15 \
--affine_scale 0.2 \
--input_shape 104 104 92 \
--batch_size 4 \
--lr 3e-5 \
--save_frequency 20 \
--num_epochs 10000 \
--using_HPC 1 
                   
                   
