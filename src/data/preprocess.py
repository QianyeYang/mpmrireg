import nibabel as nib
import numpy as np

def center_crop():
    pass

def resample():
    pass

def normalize(arr, method='01'):
    return (arr - np.mean(arr)) / np.std(arr)

def gen_rand_affine_transform(batch_size, scale, seed=525):
    """
    https://github.com/DeepRegNet/DeepReg/blob/d3edf264b8685b47f1bdd9bb73aca79b1a72790b/deepreg/dataset/preprocess.py
    :param scale: a float number between 0 and 1
    :return: shape = (batch, 4, 3)
    """

    assert 0 <= scale <= 1
    np.random.seed(seed)
    noise = np.random.uniform(1 - scale, 1, [batch_size, 4, 3])  # shape = (batch, 4, 3)

    # old represents four corners of a cube
    # corresponding to the corner C G D A as shown above
    old = np.tile(
        [[[-1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]],
        [batch_size, 1, 1],
    )  # shape = (batch, 4, 4)
    new = old[:, :, :3] * noise  # shape = (batch, 4, 3)

    theta = np.array(
        [np.linalg.lstsq(old[k], new[k], rcond=-1)[0] for k in range(batch_size)]
    )  # shape = (batch, 4, 3)

    return theta

def grid_affine_transform(grid, theta):
    pass