import torch
import torch.nn.functional as F
import pystrum.pynd.ndutils as nd
import numpy as np


def ssd(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)

def single_scale_dice(y_true, y_pred, eps=1e-6):
    numerator = torch.sum(y_true * y_pred, axis=[1, 2, 3, 4]) * 2
    denominator = torch.sum(y_true, axis=[1, 2, 3, 4]) + torch.sum(y_pred, axis=[1, 2, 3, 4]) + eps
    return 1 - torch.mean(numerator / denominator)

def binary_dice(y_true, y_pred):
    eps = 1e-6
    y_true = y_true >= 0.5
    y_pred = y_pred >= 0.5
    numerator = torch.sum(y_true * y_pred) * 2
    denominator = torch.sum(y_true) + torch.sum(y_pred) + eps
    return numerator * 1.0 / denominator

def global_mutual_information(t1, t2):
    bin_centers = torch.linspace(0.0, 1.0, 21)

    if t1.is_cuda:
        bin_centers = bin_centers.cuda()

    num_bins = bin_centers.shape[0]
    sigma_ratio = 0.5
    eps = 1.19209e-07
    
    sigma = torch.mean(bin_centers[1:]-bin_centers[:-1])*sigma_ratio
    preterm = 1 / (2 * torch.square(sigma))
    
    if len(t1.shape) == 3:
        w, h, z = t1.shape
        c, batch = 1, 1
    elif len(t1.shape) == 4:
        c, w, h, z = t1.shape
        batch = 1 
    elif len(t1.shape) == 5:
        batch, c, w, h, z = t1.shape
    else:
        raise NotImplementedError 

    t1 = torch.reshape(t1, [batch, c*w*h*z, 1])
    t2 = torch.reshape(t2, [batch, c*w*h*z, 1])
    nb_voxels = t1.shape[1] * 1.0
    vbc = torch.reshape(bin_centers, (1, 1, -1))

    I_a = torch.exp(- preterm * torch.square(t1 - vbc))
    I_a = I_a / torch.sum(I_a, -1, keepdim=True)
    I_a_permute = torch.transpose(I_a, 2, 1)
    
    I_b = torch.exp(- preterm * torch.square(t2 - vbc))
    I_b = I_b / torch.sum(I_b, -1, keepdim=True)
    I_b_permute = torch.transpose(I_b, 2, 1)

    pa = torch.mean(I_a, axis=1, keepdim=True)
    pb = torch.mean(I_b, axis=1, keepdim=True)
    pa = torch.transpose(pa, 2, 1)

    papb = torch.matmul(pa, pb) + eps
    pab = torch.matmul(I_a_permute, I_b) / nb_voxels

    return torch.mean(torch.sum(pab*torch.log(pab/papb + eps), dim=[1, 2]))

    
def gradient_dx(arr):
    return (arr[:, 2:, 1:-1, 1:-1] - arr[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(arr):
    return (arr[:, 1:-1, 2:, 1:-1] - arr[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(arr):
    return (arr[:, 1:-1, 1:-1, 2:] - arr[:, 1:-1, 1:-1, :-2]) / 2


def gradient_txyz(Txyz, fn):
    return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1, 2]], axis=1)


def bending_energy(ddf):
    # 1st order
    dTdx = gradient_txyz(ddf, gradient_dx)
    dTdy = gradient_txyz(ddf, gradient_dy)
    dTdz = gradient_txyz(ddf, gradient_dz)

    # 2nd order
    dTdxx = gradient_txyz(dTdx, gradient_dx)
    dTdyy = gradient_txyz(dTdy, gradient_dy)
    dTdzz = gradient_txyz(dTdz, gradient_dz)
    dTdxy = gradient_txyz(dTdx, gradient_dy)
    dTdyz = gradient_txyz(dTdy, gradient_dz)
    dTdxz = gradient_txyz(dTdx, gradient_dz)

    return torch.mean(dTdxx ** 2 + dTdyy ** 2 + dTdzz ** 2 + 2 * dTdxy ** 2 + 2 * dTdxz ** 2 + 2 * dTdyz ** 2)

def normalized_bending_energy(ddf, voxel_size, image_shape):
    """normalize the ddf to the real physical space"""

    x, y, z = image_shape
    vx, vy, vz = voxel_size

    # 1st order - the gradient is calculated with uint /1mm
    dTdx = gradient_txyz(ddf, gradient_dx)
    dTdy = gradient_txyz(ddf, gradient_dy)
    dTdz = gradient_txyz(ddf, gradient_dz)

    dTdx = dTdx * (x/2.0) / vx
    dTdy = dTdy * (y/2.0) / vy
    dTdz = dTdz * (z/2.0) / vz

    # 2nd order
    dTdxx = gradient_txyz(dTdx, gradient_dx)
    dTdyy = gradient_txyz(dTdy, gradient_dy)
    dTdzz = gradient_txyz(dTdz, gradient_dz)
    dTdxy = gradient_txyz(dTdx, gradient_dy)
    dTdyz = gradient_txyz(dTdy, gradient_dz)
    dTdxz = gradient_txyz(dTdx, gradient_dz)

    return torch.mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2)


def l2_gradient(ddf):
    dTdx = gradient_txyz(ddf, gradient_dx)
    dTdy = gradient_txyz(ddf, gradient_dy)
    dTdz = gradient_txyz(ddf, gradient_dz)
    return torch.mean(dTdx**2 + dTdy** 2 + dTdz**2)


def mmd(x1, x2, sigmas):
    """the loss of maximum mean discrepancy."""
    x1 = torch.reshape(x1, [x1.shape[0], -1])
    x2 = torch.reshape(x2, [x2.shape[0], -1])
    # print('x1x2shape:', x1.shape)
    diff = torch.mean(gaussian_kernel(x1, x1, sigmas))  # mean_x1x1
    diff -= 2 * torch.mean(gaussian_kernel(x1, x2, sigmas))  # mean_x1x2
    diff += torch.mean(gaussian_kernel(x2, x2, sigmas))  # mean_x2x2
    # print('diff:', diff, diff.shape)
    return diff

def gaussian_kernel(x1, x2, sigmas):
    beta = 1. / (2. * (torch.unsqueeze(sigmas, dim=1)))
    # print('beta shape:', beta.shape)
    dist = torch.sum(torch.square(torch.unsqueeze(x1, dim=2) - torch.transpose(x2, 1, 0)), dim=1)
    # print('dist:', dist.shape)
    dist = torch.transpose(dist, 1, 0)
    s = torch.matmul(beta, torch.reshape(dist, (1, -1)))

    k = torch.exp(-s)
    
    # print('k:', k, k.shape )

    mmd = torch.sum(k, dim=0)
    # print('mmd:', mmd, mmd.shape)
    return torch.reshape(mmd, dist.shape)

# def gaussian_kernel(x1, x2, sigmas):
#     beta = 1. / (2. * (torch.unsqueeze(sigmas, dim=1)))
#     dist = torch.sum(torch.square(torch.unsqueeze(x1, dim=2) - torch.transpose(x2, 1, 0)), dim=1)
#     dist = torch.transpose(dist, 1, 0)
#     s = torch.matmul(beta, torch.reshape(dist, (1, -1)))

#     mmd = torch.sum(torch.exp(-s), dim=0)
#     print('mmd:', mmd, mmd.shape )
#     return torch.reshape(mmd, dist.shape)

def compute_centroid(mask):
    assert torch.sum(mask) > 0, 'nothing find on the mask'
    mask = mask >= 0.5  # shape (1, 1, x, y, z)
    mesh_points = [torch.tensor(list(range(dim))) for dim in mask.shape[2:]]
    grid = torch.stack(torch.meshgrid(*mesh_points))  # shape:[3, x, y, z]
    grid = grid.type(torch.FloatTensor)
    grid = grid.cuda()
    grid = grid * mask[0]
    return torch.sum(grid, axis=(1, 2, 3))/torch.sum(mask[0])

def centroid_distance(y_true, y_pred):
    c1 = compute_centroid(y_true)
    c2 = compute_centroid(y_pred)
    return torch.sqrt(torch.sum((c1-c2)**2))


def wBCE(output, target, weights=None):
    '''weighted_binary_cross_entropy loss'''       
    output = torch.clamp(output,min=1e-6,max=1-1e-6)

    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))  


def jacobian_determinant(disp):
    '''disp shape : [b, 3, x, y, z]'''

    # check inputs
    disp = torch.squeeze(disp)
    disp = disp.permute(1,2,3,0)
    disp = disp.cpu().numpy()

    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else: # must be 2 
        
        dfdx = J[0]
        dfdy = J[1] 
        
        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]