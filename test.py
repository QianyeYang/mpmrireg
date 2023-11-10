import os
from src.model import archs
from config.config_utils import print_config
import pickle as pkl
import sys

if sys.argv[1].endswith('/') or sys.argv[1].endswith('\\'):
    sys.argv[1] = sys.argv[1][:-1]
exp_name = os.path.basename(sys.argv[1])
project_name = os.path.basename(os.path.dirname(sys.argv[1]))

print(f'evaluation... {exp_name}')

if len(sys.argv) > 2:
    gpu_idx = sys.argv[2]
else:
    gpu_idx = '0'

if len(sys.argv) > 3:
    num_epoch = int(sys.argv[3])
else:
    num_epoch = 'best'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx



if __name__ == "__main__":
    with open(os.path.join(f'./logs/{project_name}/{exp_name}/config.pkl'), 'rb') as f:
        config = pkl.load(f)
        print_config(config)

    if not hasattr(config, 'patched'):
        config.patched = 0

    if config.project == "mpmrireg":
        from src.model.archs.mpmrireg import mpmrireg
        model = mpmrireg(config)
    else:
        raise NotImplementedError

    model.load_epoch(num_epoch = num_epoch)
    model.inference()
    print('inference done.')
