import src.model.networks as networks
from src.model import loss
from src.model.functions import warp3d,ddf_merge,get_reference_grid3d
from src.model.archs.baseArch import BaseArch
from src.data import dataloaders
import torch, os
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np
from scipy import stats
from src.data.preprocess import gen_rand_affine_transform
from time import time
import pystrum.pynd.ndutils as nd


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

class weakSuperVisionMpMRIReg(BaseArch):
    def __init__(self, config):
        super(weakSuperVisionMpMRIReg, self).__init__(config)
        self.net = networks.LocalModel(input_shape=(104, 104, 92)).cuda()
        self.set_dataloader()
        self.best_metric = 0
        
    def set_dataloader(self):
        self.train_set = dataloaders.mpMRIData(config=self.config, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        print('>>> Train set ready (weakSuperVisionMpMRIReg).')
        self.val_set = dataloaders.mpMRIData(config=self.config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        print('>>> Validation set ready (weakSuperVisionMpMRIReg).')
        self.test_set = dataloaders.mpMRIData(config=self.config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        print('>>> Holdout set ready (weakSuperVisionMpMRIReg).')

    def get_input(self, input_dict, mv_mod):
        if mv_mod == 'dwi':
            fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()  # [batch, 1, x, y, z]
        elif mv_mod == 'mixed':
            fx_img, mv_img = input_dict['t2'].cuda(), input_dict['mv_img'].cuda()
        else:
            raise NotImplementedError
        weakly_img = input_dict['dwi_b0'].cuda()
        return fx_img, mv_img, weakly_img

    def train(self):
        self.save_configure()
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        for self.epoch in range(1, self.config.num_epochs + 1):
            self.net.train()
            for self.step, input_dict in enumerate(self.train_loader):
                fx_img, mv_img, weakly_img = self.get_input(input_dict, self.config.mv_mod)

                if self.config.random_affine_augmentation:
                    affine_grid = self.rand_affine_grid(mv_img)
                    affine_grid = affine_grid.permute(0, 2, 3, 4, 1)
                    affine_grid = affine_grid[..., [2, 1, 0]]
                    mv_img = torch.nn.functional.grid_sample(mv_img, affine_grid, mode='bilinear', align_corners=False)

                optimizer.zero_grad()
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_weakly_img = warp3d(weakly_img, ddf)
                warpped_img = warp3d(mv_img, ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img, warpped_weakly_img)
                global_loss.backward()
                optimizer.step()
            if self.epoch % self.config.save_frequency == 0:
                self.save()
            print('-'*10, 'validation', '-'*10)
            self.validation(dataloader=self.val_loader)

    def loss(self, ddf, fx_img, wp_img, wp_wk_img):
        L_Dreg_bde = loss.bending_energy(ddf) * self.config.w_bde
        L_Dreg_l2n = loss.l2_gradient(ddf) * self.config.w_l2n
        L_Isim = (1.5 - loss.global_mutual_information(fx_img, wp_img)) * self.config.w_gmi * self.config.w_uns
        L_Isim_wk = (1.5 - loss.global_mutual_information(fx_img, wp_wk_img)) * self.config.w_gmi * self.config.w_wks
        L_All = L_Dreg_bde + L_Dreg_l2n + L_Isim + L_Isim_wk
        Info = f'epoch {self.epoch}, step {self.step+1}, L_All:{L_All:.3f}, Loss_Dreg_bde: {L_Dreg_bde:.3f}, Loss_Dreg_l2n: {L_Dreg_l2n:.3f}, Loss_Isim: {L_Isim:.3f}, Loss_Isim_wk: {L_Isim_wk:.3f}'
        print(Info)
        return L_All

    @torch.no_grad()
    def validation(self, dataloader):
        self.net.eval()
        res = []
        for idx, input_dict in enumerate(dataloader):
                fx_img, mv_img, weakly_img = self.get_input(input_dict, mv_mod='dwi')
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_img = warp3d(mv_img, ddf)
                warpped_weakly_img = warp3d(weakly_img, ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img, warpped_weakly_img)
                res.append(loss.global_mutual_information(fx_img, warpped_img))
        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)
        if mean > self.best_metric:
            self.best_metric = mean
            self.save(type='best')
        print('Result:', mean, std)

    def rand_affine_grid(self, img):
        grid = get_reference_grid3d(img)  #(batch, 4, 3) (b, i, j)
        theta = gen_rand_affine_transform(
            img.shape[0], self.config.test_affine_scale, seed=self.config.test_affine_seed)  # [batch, 3, x, y, z]  (b, j, x, y, z)
        theta = torch.FloatTensor(theta).cuda()
        padded_grid = torch.cat([grid, torch.ones_like(grid[:, :1, ...])], axis=1)
        warpped_grids = torch.einsum('bixyz,bij->bjxyz', padded_grid, theta)
        # return warp3d(img, warpped_grid), warp3d(label, warpped_grid)
        return warpped_grids

    @torch.no_grad()
    def inference(self):
        if self.config.test_with_affine: print("NOTE: random affine is used in the test!")
        self.net.eval()
        visualization_path = os.path.join(self.log_dir, f'vis-{self.epoch}')
        os.makedirs(visualization_path, exist_ok=True)
        
        results = {'mi':[], 'ldmk':[], 'mi-wo-reg':[], 'ldmk-wo-reg':[]}
        infos = {'ldmk_paths':[]}
        ldmk_improvement_topX = []
        ldmk_num_each_sample = []
        ldmk_footprint = []

        # total_time = 0
        # start = time()
        for idx, input_dict in enumerate(self.test_loader):
            
            fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()  # [batch, 1, x, y, z]

            if self.config.test_with_affine:
                affine_grid = self.rand_affine_grid(mv_img)
                affine_grid = affine_grid.permute(0, 2, 3, 4, 1)
                affine_grid = affine_grid[..., [2, 1, 0]]
                mv_img = torch.nn.functional.grid_sample(mv_img, affine_grid, mode='bilinear', align_corners=False)

            
            _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
            warpped_img = warp3d(mv_img, ddf)  # [batch=1, 1, w, h, z]
            warpped_pr_img = warp3d(input_dict['dwi_b0'].cuda(), ddf)
            results['mi'].append(loss.global_mutual_information(fx_img, warpped_img).cpu().numpy())
            results['mi-wo-reg'].append(loss.global_mutual_information(fx_img, mv_img).cpu().numpy())
            
            jc = jacobian_determinant(ddf)
            jc = (jc>0).all()

            t2_ldmks = input_dict['t2_ldmks'].cuda()
            dwi_ldmks = input_dict['dwi_ldmks'].cuda()

            ldmk_num_each_sample.append(t2_ldmks.shape[1])

            for i in range(t2_ldmks.shape[1]):
                ldmk_footprint.append(idx)
                t2ld = t2_ldmks[:, i:i+1, :, :, :]
                dwild = dwi_ldmks[:, i:i+1, :, :, :]

                if self.config.test_with_affine:
                    affine_grid = self.rand_affine_grid(dwild)
                    affine_grid = affine_grid.permute(0, 2, 3, 4, 1)
                    affine_grid = affine_grid[..., [2, 1, 0]]
                    dwild = torch.nn.functional.grid_sample(dwild, affine_grid, mode='bilinear', align_corners=False)

                wpld = warp3d(dwild, ddf)
                rs = loss.centroid_distance(t2ld, wpld).cpu().numpy()
                rs_wo_reg = loss.centroid_distance(t2ld, dwild).cpu().numpy()
                if not np.isnan(rs):
                    results['ldmk'].append(rs)
                    results['ldmk-wo-reg'].append(rs_wo_reg)
                    t2_ldmk_path = input_dict['t2_ldmks_paths'][i][0]
                    dwi_ldmk_path = input_dict['dwi_ldmks_paths'][i][0]
                    t2_pid, t2_ldmk_file = t2_ldmk_path.split('/')[-2], t2_ldmk_path.split('/')[-1]
                    dwi_pid, dwi_ldmk_file = dwi_ldmk_path.split('/')[-2], dwi_ldmk_path.split('/')[-1]
                    assert t2_pid == dwi_pid, print(f"pid not match, t2:{t2_pid}, dwi:{dwi_pid}")
                    infos['ldmk_paths'].append((t2_ldmk_path, dwi_ldmk_path))
                    print(
                        f'{idx+1}-{i+1}',
                        t2_pid,
                        t2_ldmk_file, dwi_ldmk_file, 
                        'woreg:', np.around(rs_wo_reg, decimals=3), 
                        'after-reg:', np.around(rs, decimals=3), 
                        'ipmt:', np.around(rs_wo_reg - rs, decimals=3),
                        'jacobian:', jc
                        )
                    ldmk_improvement_topX.append(rs_wo_reg - rs)
                else:
                    print(i+1, 'warning: nan exists.')
            
            print('-' * 20)
            self.save_img(fx_img, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            self.save_img(mv_img, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))
            self.save_img(input_dict['dwi_b0'].cuda(), os.path.join(visualization_path, f'{idx+1}-pr_img.nii'))
            self.save_img(warpped_pr_img, os.path.join(visualization_path, f'{idx+1}-wp_pr_img.nii'))
            self.save_img(warpped_img, os.path.join(visualization_path, f'{idx+1}-wp_img.nii'))
            self.save_img(ddf[0, 0, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-x.nii'))
            self.save_img(ddf[0, 1, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-y.nii'))
            self.save_img(ddf[0, 2, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-z.nii'))
        
        """check the improvement on the worst cases"""
        case_num = 14
        wo_reg_topX, after_reg_topX = [], []
        dist_copy = results['ldmk-wo-reg'].copy()

        misalignment_footprint = [i for _, i in sorted(zip(dist_copy, ldmk_footprint), reverse=True)]
        dist_copy.sort(reverse=True)

        showed_up, bef_mis, aft_mis = [], [], []
        for idx in misalignment_footprint[:case_num]:
            if idx not in showed_up:
                showed_up.append(idx)
                bef_mis.append(results['mi-wo-reg'][idx])
                aft_mis.append(results['mi'][idx])
            else:
                continue

        ldmk_improvement_bef = [i for _, i in sorted(zip(ldmk_improvement_topX, results['ldmk-wo-reg'].copy()), reverse=True)]
        ldmk_improvement_aft = [i for _, i in sorted(zip(ldmk_improvement_topX, results['ldmk'].copy()), reverse=True)]
        improvement_footprint = [i for _, i in sorted(zip(ldmk_improvement_topX, ldmk_footprint), reverse=True)]
        ldmk_improvement_topX.sort(reverse=True)

        standard = dist_copy[case_num]
        for (wr_dist, ar_dist) in zip(results['ldmk-wo-reg'], results['ldmk']):
            if wr_dist > standard:
                wo_reg_topX.append(wr_dist)
                after_reg_topX.append(ar_dist)

        print(f'wo_reg_top_{case_num}_tre:', np.around(np.mean(wo_reg_topX), decimals=3), np.around(np.std(wo_reg_topX), decimals=3))
        print(f'after_reg_top_{case_num}_tre:', np.around(np.mean(after_reg_topX), decimals=3), np.around(np.std(after_reg_topX), decimals=3))
        print(f'wo_reg_top_{case_num}_mi:', np.around(np.mean(bef_mis), decimals=3), np.around(np.std(bef_mis), decimals=3))
        print(f'after_reg_top_{case_num}_mi:', np.around(np.mean(aft_mis), decimals=3), np.around(np.std(aft_mis), decimals=3))
        print('p-value-tre:', stats.ttest_rel(after_reg_topX, wo_reg_topX)[1])
        print('p-value-mi:', stats.ttest_rel(aft_mis, bef_mis)[1])
        #######
        with open(os.path.join(self.log_dir, f'misalignment_top_{case_num}.pkl'), 'wb') as f:
            misalignment_res = {'ldmk':after_reg_topX, 'mi':aft_mis}
            pkl.dump(misalignment_res, f)

        print(f'improvement_top_{case_num}:', np.around(np.mean(ldmk_improvement_topX[:case_num]), decimals=3), np.around(np.std(ldmk_improvement_topX[:case_num]), decimals=3))
        print(f'improvement_top{case_num}_bef-tre:', np.around(np.mean(ldmk_improvement_bef[:case_num]), decimals=3), np.around(np.std(ldmk_improvement_bef[:case_num]), decimals=3))
        print(f'improvement_top{case_num}_aft-tre:', np.around(np.mean(ldmk_improvement_aft[:case_num]), decimals=3), np.around(np.std(ldmk_improvement_aft[:case_num]), decimals=3))
        top_pids = list(set(improvement_footprint[:case_num]))
        bef_mis = [results['mi-wo-reg'][i] for i in top_pids]
        aft_mis = [results['mi'][i] for i in top_pids]
        print(f'improvement_top{case_num}_bef-mi:', np.around(np.mean(bef_mis), decimals=3), np.around(np.std(bef_mis), decimals=3))
        print(f'improvement_top{case_num}_aft-mi:', np.around(np.mean(aft_mis), decimals=3), np.around(np.std(aft_mis), decimals=3))
        print('p-value:', stats.ttest_rel(ldmk_improvement_bef[:case_num], ldmk_improvement_aft[:case_num])[1])
        print('p-value-mi:', stats.ttest_rel(aft_mis, bef_mis)[1])

        tmp_dict = {}
        for k, v in results.items():
            mean, std = np.around(np.mean(v), decimals=3), np.around(np.std(v), decimals=3)
            print(k, mean, std)
            tmp_dict[f'{k}_stat'] = mean, std
        results.update(tmp_dict)

        results['mi-p-value'] = stats.ttest_rel(results['mi'], results['mi-wo-reg'])[1]
        results['ldmk-p-value'] = stats.ttest_rel(results['ldmk'], results['ldmk-wo-reg'])[1]
        print('mi-p-value:', results['mi-p-value'])
        print('ldmk-p-value:', results['ldmk-p-value'])

        # --------------------------percentiles----------------------------
        print('-----90 percentiles----')
        print('before-reg, TRE', np.around(np.percentile(results['ldmk-wo-reg'], 90), decimals=3))
        print('after-reg, TRE', np.around(np.percentile(results['ldmk'], 90), decimals=3))
        print('before-reg, MI', np.around(np.percentile(results['mi-wo-reg'], 90), decimals=3))
        print('after-reg, MI', np.around(np.percentile(results['mi'], 90), decimals=3))
        print('-----medians----')
        print('before-reg, TRE', np.around(np.median(results['ldmk-wo-reg']), decimals=3))
        print('after-reg, TRE', np.around(np.median(results['ldmk']), decimals=3))
        print('before-reg, MI', np.around(np.median(results['mi-wo-reg']), decimals=3))
        print('after-reg, MI', np.around(np.median(results['mi']), decimals=3))

        # print(f'total time used: {total_time}s, average: {total_time/len(self.test_loader)}s')
        
        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)

        with open(os.path.join(self.log_dir, 'infos.pkl'), 'wb') as f:
            pkl.dump(infos, f)

