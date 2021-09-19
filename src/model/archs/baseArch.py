import time, torch, sys, os
import torch.utils.data as tdata
from src.data import dataloaders
import nibabel as nib
from datetime import datetime
import pickle as pkl
from glob import glob
import numpy as np

class BaseArch(object):
    def __init__(self, config):
        """basic settings"""
        self.config = config
        self.log_dir = self.get_log_dir()
        
        """to be set in children obj"""
        self.net = None

        """global variables"""
        self.epoch, self.step = 0, 0
        self.best_model = ''
        self.global_step = 0
        self.global_epoch = 0
        self.epoch_loss = 0

    """define in children obj"""
    def train(self):
        pass
    def validate(self):
        pass
    def evaluation(self):
        pass
    def loss(self):
        pass
    def set_dataloader(self):
        pass

    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('>>> Using GPU.')
        else:
            device = torch.device('cpu')
            print('>>> Using CPU')
        return device

    def save(self, type=None):
        ckpt_path = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)
        if type is None:
            torch.save(self.net, os.path.join(ckpt_path, f'epoch-{self.epoch}.pt'))
        elif type == 'best':
            exist_best_models = glob(os.path.join(ckpt_path, 'best*.pt'))
            [os.remove(i) for i in exist_best_models]
            torch.save(self.net, os.path.join(ckpt_path, f'best-epoch-{self.epoch}.pt'))
        else:
            pass

    def load_epoch(self, num_epoch):
        if num_epoch != 'best':
            self.epoch = int(num_epoch)
            self.net = torch.load(os.path.join(self.log_dir, 'checkpoints', f'epoch-{num_epoch}.pt'))
        else:
            self.epoch = num_epoch
            best_ckpt = glob(os.path.join(self.log_dir, 'checkpoints', f'best*'))
            assert(len(best_ckpt)) != 0, "no best ckpt found in this exp..."
            self.net = torch.load(best_ckpt[0])

    def save_configure(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'config.pkl'), 'wb') as f:
            pkl.dump(self.config, f)

    def get_log_dir(self):
        assert self.config.exp_name is not None, "exp_name should not be None."
        log_dir = os.path.join(self.config.log_path, self.config.exp_name)
        while os.path.exists(log_dir) and 'train.py' in sys.argv[0]:
            log_dir = os.path.join(self.config.log_path, self.config.exp_name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        return log_dir

    @staticmethod
    def save_img(tensor_arr, save_path):
        save_folder = os.path.dirname(save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        arr = torch.squeeze(tensor_arr)
        assert len(arr.shape)==3, "not a 3 dimentional volume, need to check."
        arr = arr.cpu().numpy()
        nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
        nib_img.header['pixdim'][1:4] = np.array([1.0, 1.0, 1.0])
        nib.save(img=nib_img, filename=save_path)
