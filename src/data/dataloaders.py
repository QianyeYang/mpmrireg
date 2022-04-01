import torch.utils.data as data
import pickle as pkl
import os
import torch
import random
import numpy as np
from glob import glob
from src.model.loss import global_mutual_information
import src.model.functions as smfunction
import src.data.preprocess as pre

torch.set_default_tensor_type('torch.FloatTensor')


class mpMRIData(data.Dataset):
    def __init__(self, config, phase, external=None):
        assert phase in ['train', 'val', 'test'], "phase incorrect..."
        self.phase = phase
        self.config = config
        self.is_external = False

        if (external is not None) and phase=='test':
            self.is_external = True
            self.data_path = external
            self.data_list = glob(os.path.join(self.data_path, '*'))
        else:
            self.data_path = self.config.data_path
            self.data_list = glob(os.path.join(self.data_path, self.phase, '*'))

        self.data_list.sort()
        self.mi_record_list = self.data_list.copy()

    def __getitem__(self, index):
        data = {}
        image_path = self.data_list[index]
        mod = ['t2', 'dwi_b0', 'dwi']
        for m in mod:
            if self.is_external and m=='dwi_b0':
                matrix = np.load(os.path.join(image_path, 'dwi.npy'))
                data[m] = torch.FloatTensor(self.normalize(matrix[None, ...]))
                data[f'{m}_path'] = os.path.join(image_path, 'dwi.npy')
                continue
            matrix = np.load(os.path.join(image_path, f'{m}.npy'))
            data[m] = torch.FloatTensor(self.normalize(matrix[None, ...]))
            data[f'{m}_path'] = os.path.join(image_path, f'{m}.npy')

        if self.phase=='train' and self.config.method=='mixed':
            '''0.5 probability to use dwi_b0 and 0.5 for dwi as moving image'''
            if self.rand_prob(0.5):
                # print('mixed mode')
                data['dwi'] = torch.clone(data['dwi_b0'])

        if self.phase == 'test':
            data.update(self.__get_landmarks__(image_path))
        return data

    def __get_landmarks__(self, image_path):
        t2_ldmk_paths = glob(os.path.join(image_path, 't2_ldmk_*'))
        dwi_ldmk_paths = [i.replace('t2_ldmk_', 'dwi_ldmk_') for i in t2_ldmk_paths]
        
        t2_ldmk_paths.sort()
        dwi_ldmk_paths.sort()

        t2_ldmks = [self.normalize(np.load(i)) for i in t2_ldmk_paths]
        t2_ldmks = torch.FloatTensor(np.stack(t2_ldmks))
        dwi_ldmks = [self.normalize(np.load(i)) for i in dwi_ldmk_paths]
        dwi_ldmks = torch.FloatTensor(np.stack(dwi_ldmks))
        return {'t2_ldmks': t2_ldmks, 'dwi_ldmks':dwi_ldmks, 't2_ldmks_paths':t2_ldmk_paths, 'dwi_ldmks_paths':dwi_ldmk_paths}

    def __dump_mi_record__(self, idx, tensor_arr, ori_patient_path):
        pid = os.path.basename(ori_patient_path)
        dataset_name = os.path.basename(self.config.data_path)
        tmp_dataset_name = f"{dataset_name}-{self.config.exp_name}"
        save_folder = os.path.join(os.path.dirname(self.config.data_path), tmp_dataset_name, pid)
        os.makedirs(save_folder, exist_ok=True)
        self.mi_record_list[idx] = save_folder  # update new path for transformed dwi_b0
        
        save_name = os.path.join(save_folder, 'dwi_b0.npy')
        print(save_name)
        np.save(save_name, torch.squeeze(tensor_arr).numpy())

    @staticmethod
    def normalize(arr):
        '''normalize to 0-1'''
        return (arr - arr.min())/(arr.max() - arr.min())

    @staticmethod
    def rand_prob(p=0.5):
        assert 0<=p<=1, "p should be a number in [0, 1]"
        return random.random() < p
        
    def __len__(self):
        return len(self.data_list)
      
        
