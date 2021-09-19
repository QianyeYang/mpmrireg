import os
import nibabel as nib
from glob import glob
import re

class ASData(object):
    def __init__(self, patient_id, extra_label=False, landmarks=False):
        self.data_root = '/media/yipeng/data/ActivateSurveillance/AS-Data/'
        self.patient_id = patient_id
        self.patient_folder = os.path.join(self.data_root, 'Patient{}'.format(patient_id))
        self.visit_folders = self.__get_visit_folders__()
        self.visit_times = len(self.visit_folders)
        self.t2_collections, self.label_collections, self.landmark_collections = \
            self.__check_t2_and_label_exits__(extra_label, landmarks)

    def __get_visit_folders__(self):
        """
        Get sorted visit folder paths to avoid potential bugs.
        """
        vfolders = glob(os.path.join(self.patient_folder, '*'))
        print(vfolders)
        return sorted(vfolders, key=lambda x: self.__sort_keys__(x))

    def __sort_keys__(self, folder_path):
        """
        e.g. input a fold_path like .../Visit2, then return int 2.
        """
        folder = os.path.basename(folder_path)
        key = int(re.findall('Visit(\d+)', folder)[0])
        return key

    def __check_t2_and_label_exits__(self, extra_label=False, landmarks=False):
        t2_collections, label_collections, landmarks_collections = [], [], []
        for f in self.visit_folders:
            t2_folder = os.path.join(f, 'T2')
            if os.path.exists(t2_folder):
                folder_content = os.listdir(t2_folder)
            else:
                folder_content = []
            nii_files = [i for i in folder_content if i.endswith('.nii') or i.endswith('.nii.gz')]
            if 'ProstateBoundingBox.nii' in nii_files:
                label_dir = os.path.join(t2_folder, 'ProstateBoundingBox.nii')
            elif extra_label and ('ProstateBoundingBox-1.nii.gz' in nii_files):
                label_dir = os.path.join(t2_folder, 'ProstateBoundingBox-1.nii.gz')
            else:
                label_dir = None
            if landmarks and ('landmarks.nii.gz' in nii_files):
                landmarks_dir = os.path.join(t2_folder, 'landmarks.nii.gz')
            else:
                landmarks_dir = None
            label_collections.append(label_dir)
            landmarks_collections.append(landmarks_dir)
            selected_images = [i for i in nii_files if 't2' in os.path.basename(i).lower()]
            selected_images = [i for i in selected_images if 'cor' not in i.lower()]
            selected_images = [i for i in selected_images if 'cornal' not in i.lower()]
            selected_images = [i for i in selected_images if 'sag' not in i.lower()]
            if len(selected_images) == 0:
                t2_collections.append(None)
            elif len(selected_images) == 1:
                t2_collections.append(os.path.join(t2_folder, selected_images[0]))
            else:
                print(f'multi possible T2 files found in patient{self.patient_id}, {f}')
                raise(NotImplementedError)

        if not landmarks:
            return t2_collections, label_collections, []
        else:
            return t2_collections, label_collections, landmarks_collections

    def check_selection(self, t2=True, label=False, landmarks=False):
        t2_cont, label_cont, landmarks_cont = False, False, False
        if t2:
            t2_idxs = [idx for idx, i in enumerate(self.t2_collections) if i != []]
            if [] in self.t2_collections:
                print(self.patient_id, 't2 not continuous', t2_idxs, 'max-vis-time:', self.visit_times)
            else:
                print(self.patient_id, 't2 continuous', t2_idxs, 'max-vis-time:', self.visit_times)
                t2_cont = True
        if label:
            label_idxs = [idx for idx, i in enumerate(self.label_collections) if i is not None]
            if None in self.label_collections:
                print(self.patient_id, 'label not continuous', label_idxs, 'max-vis-time:', self.visit_times)
            else:
                print(self.patient_id, 'label continuous', label_idxs, 'max-vis-time:', self.visit_times)
                label_cont = True
        if landmarks:
            landmarks_idxs = [idx for idx, i in enumerate(self.landmark_collections) if i is not None]
            if None in self.landmark_collections:
                print(self.patient_id, 'landmarks not continuous', landmarks_idxs, 'max-vis-time:', self.visit_times)
            else:
                print(self.patient_id, 'landmarks continuous', landmarks_idxs, 'max-vis-time:', self.visit_times)
                landmarks_cont = True
        if not landmarks:
            return t2_cont, t2_idxs, label_cont, label_idxs
        else:
            return t2_cont, t2_idxs, label_cont, label_idxs, landmarks_cont, landmarks_idxs

    def get_image_data_4AS(self, visit_number, modality, request_data=True, normalize=False, resample=None):
        """
        if request_data is True, return image data and path,
        or only the path will be returned.
        modality is T2 or DWI
        if use resample, set this param....
        """
        image_folder = os.path.join(self.patient_folder, 'Visit{}'.format(visit_number), modality)
        if modality == 'T2':
            image_path = glob(os.path.join(image_folder, '*SFOV_TE*'))
        elif modality == 'DWI':
            image_path = glob(os.path.join(image_folder, '*diff_new_16_measipat_ADC.nii'))
        else:
            print('modality wrong and does not exist.')

        if len(image_path) == 0:  # indicates no corresponding image
            return None

        assert len(image_path) <= 1, '{} image is not unique'.format(modality)
        image_path = image_path[0]

        if request_data == False:
            return image_path
        else:
            nibfile = nib.load(image_path)
            data = nibfile.get_data()
            data = self.__center_crop__(data)
            data = data if resample is None else self.__resample__(data, resample)
            data = normalize.zo_norm(data) if normalize else data
            return image_path, nibfile, data

    def get_selected_t2_img(self, visit_number, o_dim=[0.4, 0.4, 1.5], rad=[128, 128, 16], return_label=False,
                            return_landmarks=False):
        nib_file = nib.load(os.path.join(self.visit_folders[visit_number], 'T2', self.t2_collections[visit_number][0]))
        image_arr = nib_file.get_data()
        in_pixdim = nib_file.header['pixdim'][1:4]
        print('img_arr_shape', image_arr.shape)
        processed_data = resample.resample_space(image_arr, in_pixdim=in_pixdim, out_pixdim=o_dim)
        print('resample_shape', processed_data.shape)
        processed_data = normalize.zo_norm(processed_data, rm_percentile=95)
        processed_data = crop.center_crop(processed_data, radius=rad)
        key = 'Patient{}-Visit{}'.format(self.patient_id, visit_number)
        if return_label:
            if self.label_collections[visit_number] is None:
                label = None
            else:
                label_nibfile = self.label_collections[visit_number]
                label_arr = medImg.get_arr(label_nibfile)
                in_pixdim = medImg.get_pixdim(label_nibfile)
                processed_label = resample.resample_space(label_arr, in_pixdim=in_pixdim, out_pixdim=o_dim, order=2)
                label = crop.center_crop(processed_label, radius=rad)
        if return_landmarks:
            if self.landmark_collections[visit_number] is None:
                landmarks = None
            else:
                landmarks_nibfile = self.landmark_collections[visit_number]
                landmarks_arr = medImg.get_arr(landmarks_nibfile)
                in_pixdim = medImg.get_pixdim(landmarks_nibfile)
                landmarks = []
                for lb in range(1, np.unique(landmarks_arr).shape[0]):
                    sub_ldmk_arr = ((landmarks_arr == lb)*1).astype('int')
                    processed_landmarks = resample.resample_space(sub_ldmk_arr, in_pixdim=in_pixdim, out_pixdim=o_dim, order=2)
                    processed_landmarks = crop.center_crop(processed_landmarks, radius=rad)
                    landmarks.append(processed_landmarks)
        if (not return_label) and (not return_landmarks):
            return key, processed_data
        elif return_label and (not return_landmarks):
            return key, processed_data, label
        elif return_label and return_landmarks:
            return key, processed_data, label, landmarks
        else:
            print('something wrong...')

    def get_bgrm_t2_and_label(self, visit_number, o_dim=[0.4, 0.4, 1.5], rad=[128, 128, 16]):
        key = 'Patient{}-Visit{}'.format(self.patient_id, visit_number)
        assert self.label_collections[visit_number] is not None, f"{key} doesn't have label"
        label_nib_file = self.label_collections[visit_number]
        label_arr = medImg.get_arr(label_nib_file)
        in_pixdim = medImg.get_pixdim(label_nib_file)
        bbox = morphlogy.bboxND(label_arr)
        bbox_label = label_arr[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        bbox_label = resample.resample_space(bbox_label, in_pixdim, out_pixdim=o_dim, order=2)
        cropped_label = crop.center_crop(bbox_label, radius=rad, allow_padding=True)

        t2_nib_file = os.path.join(self.visit_folders[visit_number], 'T2', self.t2_collections[visit_number][0])
        t2_arr = medImg.get_arr(t2_nib_file) * label_arr
        bbox_t2 = t2_arr[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        bbox_t2 = resample.resample_space(bbox_t2, in_pixdim, out_pixdim=o_dim)
        cropped_t2 = crop.center_crop(bbox_t2, radius=rad, allow_padding=True)

        return key, cropped_t2, cropped_label