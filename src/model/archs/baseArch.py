import time, torch, sys, os
import nibabel as nib
import pickle as pkl
import numpy as np
from datetime import datetime
from glob import glob
import cv2
import matplotlib.pyplot as plt 

class BaseArch(object):
    def __init__(self, config):
        """basic settings"""
        self.config = config
        self.log_dir = self.get_log_dir()
        
        """to be set in children obj"""
        self.net = None

        """global variables"""
        self.epoch, self.step = 0, 0
        self.phase = 'train'
        self.best_model = ''
        self.global_step = 0
        self.global_epoch = 0
        self.epoch_loss = 0
        # self.check_gpu_info()

    """define in children obj"""
    def train(self):
        pass
    def validate(self):
        pass
    def inference(self):
        pass
    def loss(self):
        pass
    def set_dataloader(self):
        pass

    def train_mode(self):
        self.phase = 'train'
        self.net.train()

    def val_mode(self):
        self.phase = 'val'
        self.net.eval()

    def test_mode(self):
        self.phase = 'test'
        self.net.eval()

    def check_gpu_info(self):
        '''will be useful when computing on HPC :) '''
        gpu_id = torch.cuda.current_device()
        gpu_type = torch.cuda.get_device_name(gpu_id)
        print(f'>>> Computing on GPU: {gpu_type} <<<')

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
            print(f'load from epoch {self.epoch}')
        else:
            best_ckpt = glob(os.path.join(self.log_dir, 'checkpoints', 'best*'))
            assert(len(best_ckpt)) != 0, "no best ckpt found in this exp..."
            self.net = torch.load(best_ckpt[0])
            self.epoch = int(best_ckpt[0].replace('.pt', '').split('-')[-1])
            print(f'load from best epoch {best_ckpt[0]}')

    def save_configure(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'config.pkl'), 'wb') as f:
            pkl.dump(self.config, f)

    def get_log_dir(self):
        assert self.config.exp_name is not None, "exp_name should not be None."
        log_dir = os.path.join('./logs',self.config.project ,self.config.exp_name)
        while os.path.exists(log_dir) and 'train.py' in sys.argv[0] and self.config.continue_epoch=='-1':
            log_dir = os.path.join(
                './logs', 
                self.config.project, 
                self.config.exp_name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        return log_dir

    @staticmethod
    def save_img(tensor_arr, save_path, pixdim=[1.0, 1.0, 1.0]):
        save_folder = os.path.dirname(save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        arr = torch.squeeze(tensor_arr)
        assert len(arr.shape)==3, "not a 3 dimentional volume, need to check."
        arr = arr.detach().cpu().numpy()
        nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
        nib_img.header['pixdim'][1:4] = np.array(pixdim)
        nib.save(img=nib_img, filename=save_path)
        

    def get_patch_cords_from_ref_image(self, ref_img):
        patch_size = self.config.patch_size
        inf_patch_stride_factors = self.config.inf_patch_stride_factors

        if len(ref_img.shape) > 3:
            shape = ref_img.shape[-3:]
        else: shape = np.array(ref_img.shape)

        patch_size = np.array(patch_size)
        stride = patch_size // np.array(inf_patch_stride_factors)

        iters = (shape - patch_size) // stride + 1
        coords = [np.array([x, y, z])*stride for x in range(iters[0]) for y in range(iters[1]) for z in range(iters[2])]  # left top points
        coords = [list(i) for i in coords]

        z_slice = [np.array([x, y, shape[2]-patch_size[2]])*np.array([stride[0], stride[1], 1]) for x in range(iters[0]) for y in range(iters[1])]
        z_slice = [list(i) for i in z_slice]
        x_slice = [np.array([shape[0]-patch_size[0], y, z])*np.array([1, stride[1], stride[2]]) for y in range(iters[1]) for z in range(iters[2])]
        x_slice = [list(i) for i in x_slice]
        y_slice = [np.array([x, shape[1]-patch_size[1], z])*np.array([stride[0], 1, stride[2]]) for x in range(iters[0]) for z in range(iters[2])]
        y_slice = [list(i) for i in y_slice]

        zb = [np.array([shape[0]-patch_size[0], shape[1]-patch_size[1], z])*np.array([1, 1, stride[2]]) for z in range(iters[2])]  # z bound
        zb = [list(i) for i in zb]
        xb = [np.array([x, shape[1]-patch_size[1], shape[2]-patch_size[2]])*np.array([stride[0], 1, 1]) for x in range(iters[0])]  # x bound
        xb = [list(i) for i in xb]
        yb = [np.array([shape[0]-patch_size[0], y, shape[2]-patch_size[2]])*np.array([1, stride[1], 1]) for y in range(iters[1])]  # y bound
        yb = [list(i) for i in yb]
        br = [[shape[0]-patch_size[0], shape[1]-patch_size[1], shape[2]-patch_size[2]]]

        # print(len(coords), len(xb), len(yb), len(zb))

        for ex in [zb, xb, yb, br, z_slice, x_slice, y_slice]:
            for p in ex:
                if p not in coords:
                    coords.append(p)

        return [[x, x+patch_size[0], y, y+patch_size[1], z, z+patch_size[2]] for (x, y, z) in coords]


    @staticmethod
    def vis_with_contour(fx_img, fx_seg, mv_img, mv_seg, pred_seg, save_folder, sbj_name, color=(255, 255, 0), info=''):
        """fx/mv_img/seg -> 3d volume"""
        def normalize0255(arr):
            return (arr - arr.min())*255.0 / (arr.max() - arr.min())

        def add_contours(t2, label, color):
            if len(t2.shape) != 3:
                _t2 = np.tile(t2, (3,1,1)).transpose(1, 2, 0)
            else:
                _t2 = t2
            
            _t2 = normalize0255(_t2).astype('uint8')
            _label = label.astype('uint8')
            blank = np.zeros(_t2.shape)
            contours, hierarchy = cv2.findContours(_label.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
            tmp = _t2.copy()  # ?????
            cv2.drawContours(tmp, contours, -1, color, 1)
            return tmp

        img_set = np.concatenate([mv_img, fx_img, fx_img], axis=1)
        img_set = normalize0255(img_set)
        seg_set = np.concatenate([mv_seg, fx_seg, pred_seg], axis=1)
        
        for z in range(fx_img.shape[-1]):
            img_slice = img_set[..., z]
            seg_slice = seg_set[..., z]
            contoured_slice = add_contours(img_slice, seg_slice, color=color)

            save_path = os.path.join(save_folder, sbj_name)
            os.path.makedirs(save_path, exist_ok=True)

            plt.imsave(os.path.join(save_path, f"{sbj_name}_{z}_{info}.png"), contoured_slice)
            
            

                

        
        
        
        