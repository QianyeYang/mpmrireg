import nibabel as nib
import numpy as np 


def random_crop_3d(img_arr_list, patch_size):
    '''
    random crop for all the image arrays in the list.
    image will be cropped from a random generated left top corner
    the images must have the same shape
    '''
    img_shape = img_arr_list[0].shape
    for i in img_arr_list:
        assert i.shape == img_shape, "image shapes not consistent in the list"

    ix, iy, iz = img_shape
    px, py, pz = patch_size
    
    gx, gy, gz = np.random.randint(low=[ix-px, iy-py, iz-pz], size=3)
    
    return [i[gx:gx+px, gy:gy+py, gz:gz+pz] for i in img_arr_list]
    
    
def center_crop():
    pass

def resample():
    pass

def normalize(arr, method='01'):
    return (arr - np.mean(arr)) / np.std(arr)