import os
from src.model import archs
from src.model.Others import archs as oriArchs
import pickle as pkl
import sys


if sys.argv[1].endswith('/'):
    sys.argv[1] = sys.argv[1][:-1]
exp_name = os.path.basename(sys.argv[1])
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
    with open(os.path.join(f'./logs/{exp_name}/config.pkl'), 'rb') as f:
        config = pkl.load(f)
        print(config)

    if not hasattr(config, 'fx_mod'):
        config.fx_mod = 't2'
    if not hasattr(config, 'test_with_affine'):
        config.test_with_affine = 1
    if not hasattr(config, 'test_affine_scale'):
        config.test_affine_scale = 0.15
    if not hasattr(config, 'test_affine_seed'):
        config.test_affine_seed = 514

    config.test_with_affine = 0
    config.test_affine_scale = 0.1
    
    if config.model == "origin":
        model = oriArchs.mpMRIReg(config)
    elif config.model == "weakly":
        from src.model.archs.mpMRIRegWeakSupervise import weakSuperVisionMpMRIReg
        model = weakSuperVisionMpMRIReg(config)
    elif config.model == "multi-task":
        pass
    elif config.model == "joint3":
        model = oriArchs.joint3(config)
    else:
        raise NotImplementedError

    model.load_epoch(num_epoch = num_epoch)
    # model.evaluation()
    model.inference()
    print('evaluation done.')
