import os 
import pickle as pkl
import numpy as np
import nibabel as nib 
from scipy import ndimage
from glob import glob
import shutil
from tqdm import tqdm



# preprocessing
print('preprocessing...')

with open('./cia_data_dict.pkl', 'rb') as f:
    data = pkl.load(f)

class CIA_mr_study(object):
    def __init__(self, pid, data_paths):
        self.pid = pid
        self.t2_folder, self.dwi_folder = data_paths

        self.t2_nii_path = self.get_t2_nii()
        self.dwi_nii_path = self.get_dwi_nii()

    def get_t2_nii(self):
        p = glob(os.path.join(self.t2_folder, "*.nii.gz"))
        assert len(p) == 1, f"please check {self.t2_folder}"
        return p[0]

    def get_dwi_nii(self):
        p = glob(os.path.join(self.dwi_folder, "*.nii.gz"))
        assert len(p) == 1, f"please check {self.dwi_folder}"
        return p[0]

    @staticmethod
    def rotate(arr):
        return arr[::-1, ::-1, ::-1]

    @staticmethod
    def save_nifty(arr, save_path, pixdim=[1.0, 1.0, 1.0]):
        save_folder = os.path.dirname(save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
        nib_img.header['pixdim'][1:4] = np.array(pixdim)
        nib.save(img=nib_img, filename=save_path)

    @staticmethod
    def load_nib(nib_path):
        return nib.load(nib_path).get_fdata()

    @staticmethod
    def get_pixdim(nib_path):
        return nib.load(nib_path).header['pixdim'][1:4]

    @staticmethod
    def calibration(arr):
        """fix the orient of the images"""
        return arr[:,::-1,::-1]

    @staticmethod
    def resample(arr, in_pixdim, out_pixdim, order=3):
        """care about pixel physical size"""
        factor = np.array(in_pixdim) / np.array(out_pixdim)
        return ndimage.zoom(arr, zoom=factor, order=order)
    
    @staticmethod
    def center_crop(arr, radius, allow_padding=True):
        """if 2*radius exceeded the boundaries, zero-padding will be added."""
        len_x, len_y, len_z = arr.shape
        rx, ry, rz = radius[0], radius[1], radius[2]
        if len_z < 2 * rz:
            assert allow_padding is True, "boundary exceeded, consider setting allow_padding=True"
            upd, dpd = int(np.ceil((2 * rz - len_z) / 2)), int(np.floor((2 * rz - len_z) / 2))
            arr = np.pad(arr, ((0, 0), (0, 0), (upd, dpd)), 'constant', constant_values=0)
        if len_x < 2 * rx:
            assert allow_padding is True, "boundary exceeded, consider setting allow_padding=True"
            upd, dpd = int(np.ceil((2 * rx - len_x) / 2)), int(np.floor((2 * rx - len_x) / 2))
            arr = np.pad(arr, ((upd, dpd), (0, 0), (0, 0)), 'constant', constant_values=0)
        if len_y < 2 * ry:
            assert allow_padding is True, "boundary exceeded, consider setting allow_padding=True"
            upd, dpd = int(np.ceil((2 * ry - len_y) / 2)), int(np.floor((2 * ry - len_y) / 2))
            arr = np.pad(arr, ((0, 0), (upd, dpd), (0, 0)), 'constant', constant_values=0)

        len_x, len_y, len_z = arr.shape
        cx, cy, cz = int(len_x / 2), int(len_y / 2), int(len_z / 2)

        return arr[cx - rx:cx + rx, cy - ry:cy + ry, cz - rz:cz + rz]

    def single_preprocessing(self, nii_path, order=3):
        pix_dim = self.get_pixdim(nii_path)
        print('pixdim', pix_dim)
        arr = self.load_nib(nii_path)

        if len(arr.shape)==5:
            arr = arr[:, :, :, 0, 0]
        else: pass

        arr = self.resample(arr, in_pixdim=pix_dim, out_pixdim=[1.0, 1.0, 1.0], order=order)
        arr = self.center_crop(arr, radius=[52, 52, 46])
        
        return self.rotate(arr)
    
    def process_data(self):
        t2 = self.single_preprocessing(self.t2_nii_path)
        dwi = self.single_preprocessing(self.dwi_nii_path)

        dump_path = f'./CIA-external-npy/{self.pid}'
        os.makedirs(dump_path, exist_ok=True)

        self.save_nifty(t2, os.path.join(dump_path, 't2.nii.gz'))
        self.save_nifty(dwi, os.path.join(dump_path, 'dwi.nii.gz'))
        

        
for pid, data_paths in data.items():
    if pid in ['mr-1', 'mr-15', 'mr-16']:  # do not use because of bad quality
        continue
    print(pid, data_paths)
    mrs = CIA_mr_study(pid, data_paths)
    mrs.process_data()


# merge the landmarks
print('add landmarks....')
os.system('rsync -a ./CIA_landmarks/* ./CIA-external-npy')

# convert to numpy array
print('convert to numpy array...')

src_folder = 'CIA-external-npy'

def load_nib(nib_path):
    return nib.load(nib_path).get_fdata()

def dump2npy(nib_path, npy_path):
    arr = load_nib(nib_path)
    np.save(npy_path, arr)

nii_files = glob(f'./{src_folder}/**/*.nii.gz', recursive=True)

for i in nii_files:
    save_path = i.replace('.nii.gz', '.npy')
    dump2npy(i, save_path)
    os.remove(i)  # delete the nifty image

print('Done!')
