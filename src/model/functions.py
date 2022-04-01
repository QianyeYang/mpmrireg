import torch
import torch.nn.functional as F
import numpy as np
import pystrum.pynd.ndutils as nd


def get_reference_grid3d(img, grid_size=None):
    '''
    return a 5d tensor of the grid, e.g.
    img --> (b, 1, h, w, z)
    out --> (b, 3, h, w, z)

    if grid_size is not None, then return a 3d grid with the size of grid_size
    grid_size --> (gh, gw, gz)
    '''
    if len(img.shape) > 3:
        batch = img.shape[0]
    else: 
        batch = 1
    
    shape = img.shape[-3:]
    
    if grid_size is not None:
        assert len(grid_size) == 3, "maybe not a 3d grid"
        shape = grid_size

    mesh_points = [torch.linspace(-1, 1, dim) for dim in shape]
    grid = torch.stack(torch.meshgrid(*mesh_points))  # shape:[3, x, y, z]
    grid = torch.stack([grid]*batch)  # add batch
    grid = grid.type(torch.FloatTensor)
    return grid.cuda()

def warp3d(img, ddf, ref_grid=None):
    """
    img: [batch, c, x, y, z]
    new_grid: [batch, x, y, z, 3]
    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
    """

    if ref_grid is None:
        assert img.shape[-3:] == ddf.shape[-3:], "Shapes not consistent btw img and ddf."
        grid = get_reference_grid3d(img)
    else:
        grid = ref_grid
        
    new_grid = grid + ddf  # [batch, 3, x, y, z]
    new_grid = new_grid.permute(0, 2, 3, 4, 1)
    new_grid = new_grid[..., [2, 1, 0]]
    return F.grid_sample(img, new_grid, mode='bilinear', align_corners=False)

def ddf_merge(ddf_A, ddf_B):
    '''merge 2 DDFs to an equal DDF'''
    assert ddf_A.shape == ddf_B.shape, "shape of the 2 ddf must be the same"
    ref_grid = get_reference_grid3d(ddf_A)  # [batch, 3, x, y, z]
    grid_A = ref_grid + ddf_A

    grid_B = ref_grid + ddf_B
    grid_B = grid_B.permute(0, 2, 3, 4, 1)
    grid_B = grid_B[..., [2, 1, 0]]

    warped_grid_A = F.grid_sample(grid_A, grid_B, mode='bilinear', align_corners=False)  # [batch, 3, x, y, z]
    return warped_grid_A - ref_grid

def gen_rand_affine_transform(batch_size, scale, seed=None):
    """
    https://github.com/DeepRegNet/DeepReg/blob/d3edf264b8685b47f1bdd9bb73aca79b1a72790b/deepreg/dataset/preprocess.py
    :param scale: a float number between 0 and 1
    :return: shape = (batch, 4, 3)
    """
    assert 0 <= scale <= 1
    if seed is not None:
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
    

def rand_affine_grid(img, scale=0.1, random_seed=None):
    grid = get_reference_grid3d(img)  #(batch, 4, 3) (b, i, j)
    theta = gen_rand_affine_transform(img.shape[0], scale, seed=random_seed)  # [batch, 3, x, y, z]  (b, j, x, y, z)
    theta = torch.FloatTensor(theta).cuda()
    padded_grid = torch.cat([grid, torch.ones_like(grid[:, :1, ...])], axis=1)
    warpped_grids = torch.einsum('bixyz,bij->bjxyz', padded_grid, theta)

    warpped_grids = warpped_grids.permute(0, 2, 3, 4, 1)
    warpped_grids = warpped_grids[..., [2, 1, 0]]
    return warpped_grids
    


def get_reference_grid3d_cpu(img, grid_size=None):
    '''
    return a 5d tensor of the grid, e.g.
    img --> (b, 1, h, w, z)
    out --> (b, 3, h, w, z)

    if grid_size is not None, then return a 3d grid with the size of grid_size
    grid_size --> (gh, gw, gz)
    '''
    if len(img.shape) > 3:
        batch = img.shape[0]
    else: 
        batch = 1
    
    shape = img.shape[-3:]
    
    if grid_size is not None:
        assert len(grid_size) == 3, "maybe not a 3d grid"
        shape = grid_size

    mesh_points = [torch.linspace(-1, 1, dim) for dim in shape]
    grid = torch.stack(torch.meshgrid(*mesh_points))  # shape:[3, x, y, z]
    grid = torch.stack([grid]*batch)  # add batch
    grid = grid.type(torch.FloatTensor)
    return grid

def warp3d_cpu(img, ddf, ref_grid=None):
    """
    img: [batch, c, x, y, z]
    new_grid: [batch, x, y, z, 3]
    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
    """

    if ref_grid is None:
        assert img.shape[-3:] == ddf.shape[-3:], "Shapes not consistent btw img and ddf."
        grid = get_reference_grid3d_cpu(img)
    else:
        grid = ref_grid
        
    new_grid = grid + ddf  # [batch, 3, x, y, z]
    new_grid = new_grid.permute(0, 2, 3, 4, 1)
    new_grid = new_grid[..., [2, 1, 0]]
    return F.grid_sample(img, new_grid, mode='bilinear', align_corners=False)
    




