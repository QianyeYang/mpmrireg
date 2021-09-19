import torch
import torch.nn.functional as F


def get_reference_grid3d(img):
    shape = img.shape
    if len(img.shape) != 3:
        shape = shape[-3:]
    mesh_points = [torch.linspace(-1, 1, dim) for dim in shape]
    grid = torch.stack(torch.meshgrid(*mesh_points))  # shape:[3, x, y, z]
    grid = torch.stack([grid]*img.shape[0])  # add batch
    grid = grid.type(torch.FloatTensor)
    return grid.cuda()

def warp3d(img, ddf):
    """
    img: [batch, c, x, y, z]
    new_grid: [batch, x, y, z, 3]
    """
    assert img.shape[-3:] == ddf.shape[-3:], "Shapes not consistent btw img and ddf."
    grid = get_reference_grid3d(img)
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
    
    
    

    




