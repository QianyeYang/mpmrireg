import argparse
import os

parser = argparse.ArgumentParser()
# common options
parser.add_argument('--exp_name', default=None, type=str, help='experiment name you want to add.')
parser.add_argument('--data_path', default="../data/mpMriReg/FinalProcessed/52-52-46", type=str, help='the path to images')
parser.add_argument('--log_path', default="./logs", type=str, help='the key index data')
parser.add_argument('--model', default=None, type=str, help='choose model: origin/weakly/multi-task/joint')

# Training options
parser.add_argument('--lr', default=1e-5, type=float, help='Learning rate.')
parser.add_argument('--w_gmi', default=1.0, type=float, help='the weight of dice loss')
parser.add_argument('--w_bde', default=300.0, type=float, help='the weight of bending energy loss')
parser.add_argument('--w_l2n', default=50.0, type=float, help='the weight of l2 gradient loss')
parser.add_argument('--w_dsl', default=100.0, type=float, help='the weight of ddf similarity loss')
parser.add_argument('--w_uns', default=0.3, type=float, help='only works when choosing weakly model, the weight of unsupervised branch')
parser.add_argument('--w_wks', default=0.7, type=float, help='only works when choosing weakly model, the weight of weak supervise branch')
parser.add_argument('--fx_mod', default='t2', type=str, help='the modality of the fixed image, input of the network in training.')
parser.add_argument('--mv_mod', default='dwi', type=str, help='the modality of the moving image, input of the network in training.')
parser.add_argument('--mi_resample', default=0, type=int, help='only works when mv_mod==mixed, resample dwi_b0 with a affine transformation')
parser.add_argument('--mi_resample_count', default=20, type=int, help='only works when mv_mod==mixed, number of random resampling.')
parser.add_argument('--mi_resample_save', default=0, type=int, help='only works when mv_mod==mixed, save the better MI in the random affine transform?')
parser.add_argument('--use_privilege', default=0, type=int, help='use privileged image in registration')
parser.add_argument('--random_affine_augmentation', default=0, type=int, help='use random affine augmentation in training.')
parser.add_argument('--affine_scale', default=0.1, type=float, help='the scale of affine transform')




parser.add_argument('--rand_affine_aug', default=0, type=int, help='use random affine augmentation or not')
parser.add_argument('--continue_epoch', default=-1, type=int, help='continue training from a certain ckpt')  ##
parser.add_argument('--batch_size', default=4, type=int, help='The number of batch size.')
parser.add_argument('--gpu', default=0, type=int, help='id of gpu')
parser.add_argument('--num_epochs', default=3000, type=int, help='The number of iterations.')
parser.add_argument('--save_frequency', default=10, type=int, help='save frequency')
parser.add_argument('--num_channel_initial', default=16, type=int, help='number of channels in first layer.')
parser.add_argument('--ddf_levels', default=[0, 1, 2, 3, 4], nargs='+', type=int, help='ddf levels, numbers should be <= 4')
parser.add_argument('--ddf_energy_type', default="bending", type=str, help='could be gradient-l2, gradient-l1, bending')

# Testing options
parser.add_argument('--test_gen_pred_imgs', default=0, type=int, help='generate the prediction images when inference?')
parser.add_argument('--test_phase', default='test', type=str, help='use val/test set in the inference')
parser.add_argument('--test_mode', default=0, type=int, help='set a flag for test to avoid some computation')
parser.add_argument('--test_epoch', default=0, type=int, help='set a flag for test to avoid some computation')
parser.add_argument('--test_model', default=[0, 200], nargs='+', type=int, help='the range of model you wanna test.')
parser.add_argument('--test_before_reg', default=0, type=int, help='test without registration')
parser.add_argument('--test_with_affine', default=0, type=int, help='test with random affine transformation')
parser.add_argument('--test_affine_scale', default=0.1, type=float, help='scale of the affine transformation')
parser.add_argument('--test_affine_num', default=100, type=int, help='test random affine repeat number, NOT USED YET')
parser.add_argument('--test_affine_seed', default=525, type=int, help='test random affine seed, NOT USED YET')
parser.add_argument('--suffix', default='', type=str, help='the suffix for the results record')

# Others
parser.add_argument('--cv', default=0, type=int, help='cv fold number')
parser.add_argument('--repeat_num', default=100000, type=int, help='cv fold number')

args = parser.parse_args()
assert args.exp_name is not None, "experiment name should be set"
assert args.model is not None, "must select from follow: origin/weakly/multi-task/joint"