import os 
import pickle as pkl
import numpy as np
import nibabel as nib 
from scipy import ndimage
from glob import glob
import shutil
from tqdm import tqdm

with open('./CIA_selected_samples.txt', 'r') as f:
    dcm_files = f.readlines()
    dcm_files = [i.strip() for i in dcm_files]


print('copying selected data...')
for i in tqdm(dcm_files):
    target_path = i.replace("manifest-1557521391779", "Cancer_Image_Archive_Selected_Data")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(i, target_path)


# Indexing
print('indexing data....')
folders = glob('./Cancer_Image_Archive_Selected_Data/**/*-NA-*', recursive=True)

class CIA_mr_study(object):
    def __init__(self, study_path):
        self.path = study_path
        self.t2_folder, self.dwi_folder = self.get_images()

    def get_images(self):
        image_folders = os.listdir(self.path)
        t2_folder = [i for i in image_folders if "T2" in i]
        dwi_folder = [i for i in image_folders if "DWI" in i]
        assert len(t2_folder) == 1
        assert len(dwi_folder) == 1
        return os.path.join(self.path, t2_folder[0]), os.path.join(self.path, dwi_folder[0])


cia_data_dict = {}
for idx, f in enumerate(folders):
    mrs =  CIA_mr_study(f)
    key = f'mr-{idx}'
    cia_data_dict[key] = [mrs.t2_folder, mrs.dwi_folder]

with open('./cia_data_dict.pkl', 'wb') as f:
    pkl.dump(cia_data_dict, f)
