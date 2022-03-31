from . import configlib

parser = configlib.add_parser("Longitudinal registration config")

# Network options
parser.add_argument('--nc_initial', default=16, type=int, help='initial number of the channels in the frist layer of the network')
parser.add_argument('--ddf_levels', default=[0, 1, 2, 3, 4], nargs='+', type=int, help='ddf levels, numbers should be <= 4')
parser.add_argument('--inc', default=2, type=int, help='input channel number of the network')

# Training options
parser.add_argument('--model', default='LocalModel', type=str, help='LocalAffine/LocalEncoder/LocalModel')
parser.add_argument('--method', default='unsupervised', type=str, help='upsupervised/mixed/joint/privileged/B0')
parser.add_argument('--mi_resample_count', default=5, type=int, help='affine resampling time')

# loss & weights
parser.add_argument('--w_l2g', default=3000, type=float, help='the weight of the l2 gradient for the ddf.')
parser.add_argument('--w_gmi', default=0.15, type=float, help='the weight of the global mutual information.')
parser.add_argument('--w_dsl', default=1.0, type=float, help='ddf similarity loss, only used when method is joint')










