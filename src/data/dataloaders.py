import torch.utils.data as data
import pickle as pkl
import torchio as tio
import os
import torch
import random
import numpy as np
from glob import glob
from src.model.loss import global_mutual_information
from time import time
import nibabel as nib

torch.set_default_tensor_type('torch.FloatTensor')

class LongitudinalData(data.Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        self.config = config
        self.data_path = self.config.data_path
        self.key_file = os.path.join(self.data_path, self.config.key_file)
        self.key_pairs_list = self.get_key_pairs()
        self.image_folder = 'images'

    def __getitem__(self, index):
        if index == 0:
            self.key_pairs_list = self.get_key_pairs()
        moving_key, fixed_key = self.key_pairs_list[index]
        moving_image, moving_label = np.load(os.path.join(self.data_path, self.image_folder, moving_key + '-T2.npy'))
        fixed_image, fixed_label = np.load(os.path.join(self.data_path, self.image_folder, fixed_key + '-T2.npy'))
        return {
            'mv_img': torch.FloatTensor(moving_image[None, ...]), 
            'mv_seg': torch.FloatTensor(moving_label[None, ...]), 
            'fx_img': torch.FloatTensor(fixed_image[None, ...]), 
            'fx_seg': torch.FloatTensor(fixed_label[None, ...]),
            'mv_key': moving_key,
            'fx_key': fixed_key,
            }

    def __len__(self):
        return len(self.key_pairs_list)

    def get_key_pairs(self):
        '''
        have to manually define shuffling rules.
        '''
        with open(self.key_file, 'rb') as f:
            key_dict = pkl.load(f)
        l = key_dict[self.phase]
        if self.phase == 'train':
            if self.config.patient_cohort == 'intra':
                l = self.__odd_even_shuffle__(l)
            elif self.config.patient_cohort == 'inter':
                l = self.__get_inter_patient_pairs__(l)
            elif self.config.patient_cohort == 'inter+intra':
                l1 = self.__odd_even_shuffle__(l)
                l2 = self.__get_inter_patient_pairs__(l)
                l3 = self.__inter_lock__(l1, l2)
                l = l3[:len(l)]
            elif self.config.patient_cohort == 'ex+inter+intra':
                l1 = self.__odd_even_shuffle__(l)
                l2 = self.__get_inter_patient_pairs__(l, extra=key_dict['extra'])
                l3 = self.__inter_lock__(l1, l2)
                l = l3[:len(l)]
            else:
                print('wrong patient cohort.')
        return l

    def __get_inter_patient_pairs__(self, l, extra = None):
        assert 'random' in self.config.key_file, "key file should be random type"
        k = [i[0] for i in l]  # get all images
        k = list(set(k))  # get rid of repeat keys
        if extra is not None:
            assert type(extra) == list, "extra should be a list contains key values."
            k += extra
        else: pass 
        l = [(i, j) for i in k for j in k]  # get all combinations
        l = [i for i in l if i[0].split('-')[0] != i[1].split('-')[0]]  # exclude same patient
        random.shuffle(l)
        tmp = l[:len(k)]
        return tmp  # get the same length as random ordered dataloader

    @staticmethod
    def __inter_lock__(l1, l2):
        new_list = []
        for a, b in zip(l1, l2):
            new_list.append(a)
            new_list.append(b)
        return new_list

    def __odd_even_shuffle__(self, l):
        even_list, odd_list, new_list = [], [], []
        for idx, i in enumerate(l):
            if (idx % 2) == 0:
                even_list.append(i)
            else:
                odd_list.append(i)
        random.shuffle(even_list)
        random.shuffle(odd_list)
        new_list = self.__inter_lock__(even_list, odd_list)
        return new_list


class mpMRIData(data.Dataset):
    def __init__(self, config, phase):
        assert phase in ['train', 'val', 'test'], "phase incorrect..."
        self.phase = phase
        self.config = config
        self.data_path = self.config.data_path
        self.data_list = glob(os.path.join(self.data_path, self.phase, '*'))
        self.mi_record_list = self.data_list.copy()

    def __getitem__(self, index):
        data = {}
        image_path = self.data_list[index]
        mod = ['t2', 'dwi_b0', 'dwi', 'gland_mask']
        for m in mod:
            if os.path.exists(os.path.join(image_path, f'{m}_afTrans.npy')) and m=='dwi_b0' and self.config.model=='weakly' and self.config.use_privilege==1:
                matrix = np.load(os.path.join(image_path, f'{m}_afTrans.npy'))
                data[m] = torch.FloatTensor(self.normalize(matrix[None, ...]))
                data[f'{m}_path'] = os.path.join(image_path, f'{m}_afTrans.npy')
            else:
                matrix = np.load(os.path.join(image_path, f'{m}.npy'))
                data[m] = torch.FloatTensor(self.normalize(matrix[None, ...]))
                data[f'{m}_path'] = os.path.join(image_path, f'{m}.npy')

        if self.phase=='train' and self.config.mv_mod=='mixed':
            '''0.5 probability to use dwi_b0 and 0.5 for dwi as moving image'''
            if self.rand_prob(0.5):
                data['mv_img'] = torch.clone(data['dwi_b0'])
                
            else:
                if self.mi_record_list[index] != self.data_list[index]:
                    matrix = np.load(os.path.join(self.mi_record_list[index], 'dwi_b0.npy'))
                    data['dwi_b0'] = torch.FloatTensor(matrix[None, ...])

                data['mv_img'] = torch.clone(data['dwi'])
                if self.config.mi_resample:
                    affine_transform = tio.RandomAffine()
                    base_mi = global_mutual_information(data['dwi'].cuda(), data['dwi_b0'].cuda())
                    for idx in range(self.config.mi_resample_count):  # "resample b0 if can get a higher mi with hb"
                        origin_dwi_b0_matrix = np.load(os.path.join(image_path, 'dwi_b0.npy'))
                        origin_dwi_b0_matrix = torch.FloatTensor(self.normalize(origin_dwi_b0_matrix[None, ...]))
                        new_dwi_b0 = affine_transform(origin_dwi_b0_matrix)
                        mi = global_mutual_information(data['dwi'].cuda(), new_dwi_b0.cuda())
                        if mi > base_mi:
                            print('higher mi found!')
                            data['dwi_b0'] = new_dwi_b0
                            if self.config.mi_resample_save:
                                self.__dump_mi_record__(index, new_dwi_b0, image_path)
                            break

        if self.phase == 'test':
            data.update(self.__get_landmarks__(image_path))
        return data

    def __get_landmarks__(self, image_path):
        t2_ldmk_paths = glob(os.path.join(image_path, 't2_ldmk_*'))
        dwi_ldmk_paths = [i.replace('t2_ldmk_', 'dwi_ldmk_') for i in t2_ldmk_paths]
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


# def random_transform_generator(batch_size, corner_scale=.1):
#     offsets = np.tile([[[1., 1., 1.],      # np.tile([1, 4, 3], [bs, 1, 1]) --> [bs, 4, 3]
#                         [1., 1., -1.],
#                         [1., -1., 1.],
#                         [-1., 1., 1.]]],
#                       [batch_size, 1, 1]) * np.random.uniform(0, corner_scale, [batch_size, 4, 3])   # [bs, 4, 3]
#     new_corners = np.transpose(np.concatenate((np.tile([[[-1., -1., -1.],
#                                                          [-1., -1., 1.],
#                                                          [-1., 1., -1.],
#                                                          [1., -1., -1.]]],
#                                                        [batch_size, 1, 1]) + offsets,
#                                                np.ones([batch_size, 4, 1])), 2), [0, 1, 2])  # O = T I
#     src_corners = np.tile(np.transpose([[[-1., -1., -1., 1.],
#                                          [-1., -1., 1., 1.],
#                                          [-1., 1., -1., 1.],
#                                          [1., -1., -1., 1.]]], [0, 1, 2]), [batch_size, 1, 1])
#     transforms = np.array([np.linalg.lstsq(src_corners[k], new_corners[k], rcond=-1)[0]
#                            for k in range(src_corners.shape[0])])
#     transforms = np.reshape(np.transpose(transforms[:][:, :][:, :, :3], [0, 2, 1]), [-1, 1, 12])
#     return transforms

# def warp_grid(grid, theta):
#     batch_size = theta.shape[0]
#     theta = torch.FloatTensor(torch.reshape(theta, (-1, 3, 4)))
#     size = grid.get_shape().as_list()
#     grid = tf.concat([tf.transpose(tf.reshape(grid, [-1, 3])), tf.ones([1, size[0] * size[1] * size[2]])], axis=0)
#     grid = tf.reshape(tf.tile(tf.reshape(grid, [-1]), [batch_size]), [batch_size, 4, -1])
#     grid_warped = tf.matmul(theta, grid)
#     return tf.reshape(tf.transpose(grid_warped, [0, 2, 1]), [batch_size, size[0], size[1], size[2], 3])