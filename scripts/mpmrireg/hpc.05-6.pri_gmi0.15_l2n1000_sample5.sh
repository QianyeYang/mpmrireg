#$ -S /bin/bash
#$ -l tmem=12G
#$ -l gpu=true,gpu_type=titanxp
#$ -l h_rt=300:0:0
#$ -j y
#$ -N pr_m.15_g1e3_sp5
#$ -cwd
hostname
date
python3 -u train.py \
--project mpmrireg \
--exp_name 05-6.pri_gmi0.15_l2n1000_sample5 \
--data_path ./data/mpmrireg \
--method privileged \
--w_l2g 1000 \
--w_gmi 0.15 \
--affine_scale 0.2 \
--mi_resample_count 5 \
--input_shape 104 104 92 \
--batch_size 4 \
--lr 3e-5 \
--save_frequency 20 \
--num_epochs 10000 \
--using_HPC 1 
                   
                   
