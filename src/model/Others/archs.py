import src.model.networks as networks
from src.model import loss
from src.model.functions import warp3d,ddf_merge
from src.data import dataloaders
import torch
import os, sys
from datetime import datetime
import pickle as pkl
import torch.optim as optim
from torch.utils.data import DataLoader
from glob import glob
import pickle as pkl
import numpy as np
import nibabel as nib
import torchio as tio
from torchio import RandomElasticDeformation
from tqdm import tqdm
from scipy import stats

# from models.archs.baseArch import BaseArch


class mpMRIReg(object):
    def __init__(self, config):
        self.config = config
        self.net = networks.LocalModel(input_shape=(104, 104, 92)).cuda()
        self.log_dir = self.get_log_dir()
        self.train_set = dataloaders.mpMRIData(config=config, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
        self.val_set = dataloaders.mpMRIData(config=config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
        self.test_set = dataloaders.mpMRIData(config=config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
        self.best_metric = 0
        self.epoch = 0

    def get_log_dir(self):
        assert self.config.exp_name is not None, "exp_name should not be None."
        log_dir = os.path.join(self.config.log_path, self.config.exp_name)
        while os.path.exists(log_dir) and 'train.py' in sys.argv[0]:
            log_dir = os.path.join(self.config.log_path, self.config.exp_name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        return log_dir

    def save_configure(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'config.pkl'), 'wb') as f:
            pkl.dump(self.config, f)

    def train(self):
        self.save_configure()
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        for epoch in range(1, self.config.num_epochs + 1):
            self.epoch = epoch
            self.net.train()
            for idx, input_dict in enumerate(self.train_loader):
                print(f'epoch {epoch}, step {idx + 1}')
                if self.config.mv_mod in ['dwi', 'dwi_b0']:
                    fx_img, mv_img = input_dict[self.config.fx_mod].cuda(), input_dict[self.config.mv_mod].cuda()  # [batch, 1, x, y, z]
                elif self.config.mv_mod == 'mixed':
                    fx_img, mv_img = input_dict[self.config.fx_mod].cuda(), input_dict['mv_img'].cuda()  # [batch, 1, x, y, z]
                else:
                    raise NotImplementedError
                optimizer.zero_grad()
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_img = warp3d(mv_img, ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img)
                global_loss.backward()
                optimizer.step()
            if epoch % self.config.save_frequency == 0:
                self.save(epoch)
            print('-'*10, 'validation', '-'*10)
            self.validation(dataloader=self.val_loader, epoch=epoch)

    def loss(self, ddf, fx_img, wp_img):
        L_Dreg_bde = loss.bending_energy(ddf) * self.config.w_bde
        L_Dreg_l2n = loss.l2_gradient(ddf) * self.config.w_l2n
        L_Isim = (1.5 - loss.global_mutual_information(fx_img, wp_img)) * self.config.w_gmi
        L_All = L_Dreg_bde + L_Dreg_l2n + L_Isim
        Info = f'L_All:{L_All:.3f}, Loss_Dreg_bde: {L_Dreg_bde:.3f}, Loss_Dreg_l2n: {L_Dreg_l2n:.3f}, Loss_Isim: {L_Isim:.3f}'
        print(Info)
        return L_All

    def save(self, epoch, type=None):
        ckpt_path = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)
        if type is None:
            torch.save(self.net, os.path.join(ckpt_path, f'epoch-{epoch}.pt'))
        elif type == 'best':
            exist_best_models = glob(os.path.join(ckpt_path, 'best*.pt'))
            [os.remove(i) for i in exist_best_models]
            torch.save(self.net, os.path.join(ckpt_path, f'best-epoch-{epoch}.pt'))
        else:
            pass

    @torch.no_grad()
    def validation(self, dataloader, epoch=None):
        self.net.eval()
        res = []
        for idx, input_dict in enumerate(dataloader):
            if self.config.mv_mod in ['dwi', 'dwi_b0']:
                fx_img, mv_img = input_dict[self.config.fx_mod].cuda(), input_dict[self.config.mv_mod].cuda()  # [batch, 1, x, y, z]
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_img = warp3d(mv_img, ddf)
                warpped_pr_img = warp3d(input_dict['dwi_b0'].cuda(), ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img)
                res.append(loss.global_mutual_information(fx_img, warpped_img))
            elif self.config.mv_mod == 'mixed':
                fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()  # [batch, 1, x, y, z]
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_img = warp3d(mv_img, ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img)
                res.append(loss.global_mutual_information(fx_img, warpped_img))

                fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi_b0'].cuda()  # [batch, 1, x, y, z]
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_img = warp3d(mv_img, ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img)
                res.append(loss.global_mutual_information(fx_img, warpped_img))
            else:
                raise NotImplementedError
        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)
        if mean > self.best_metric:
            self.best_metric = mean
            self.save(epoch, 'best')
        print('Result:', mean, std)

    def load_epoch(self, num_epoch):
        if num_epoch != 'best':
            self.epoch = int(num_epoch)
            self.net = torch.load(os.path.join(self.log_dir, 'checkpoints', f'epoch-{num_epoch}.pt'))
        else:
            self.epoch = num_epoch
            best_ckpt = glob(os.path.join(self.log_dir, 'checkpoints', f'best*'))
            assert(len(best_ckpt)) != 0, "no best ckpt found in this exp..."
            self.net = torch.load(best_ckpt[0])

    @torch.no_grad()
    def inference(self):
        """only test on fx and mv pairs"""
        self.net.eval()
        visualization_path = os.path.join(self.log_dir, f'vis-{self.epoch}')
        os.makedirs(visualization_path, exist_ok=True)

        ldmk_footprint = []
        ldmk_improvement_topX = []
        ldmk_num_each_sample = []
        
        results = {'mi':[], 'ldmk':[], 'mi-wo-reg':[], 'ldmk-wo-reg':[]}
        for idx, input_dict in enumerate(self.test_loader):
            if self.config.mv_mod in ['dwi', 'dwi_b0']:
                fx_img, mv_img = input_dict[self.config.fx_mod].cuda(), input_dict[self.config.mv_mod].cuda()  # [batch, 1, x, y, z]
            elif self.config.mv_mod == 'mixed':
                fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()  # [batch, 1, x, y, z]
            else:
                raise NotImplementedError
            _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
            warpped_img = warp3d(input_dict['dwi'].cuda(), ddf)  # [batch=1, 1, w, h, z]
            warpped_pr_img = warp3d(input_dict['dwi_b0'].cuda(), ddf)
            results['mi'].append(loss.global_mutual_information(fx_img, warpped_img).cpu().numpy())
            results['mi-wo-reg'].append(loss.global_mutual_information(fx_img, mv_img).cpu().numpy())

            t2_ldmks = input_dict['t2_ldmks'].cuda()
            dwi_ldmks = input_dict['dwi_ldmks'].cuda()

            """calculating landmark distances"""
            for i in range(t2_ldmks.shape[1]):
                ldmk_footprint.append(idx)
                t2ld = t2_ldmks[:, i:i+1, :, :, :]
                dwild = dwi_ldmks[:, i:i+1, :, :, :]
                wpld = warp3d(dwild, ddf)
                rs = loss.centroid_distance(t2ld, wpld).cpu().numpy()
                rs_wo_reg = loss.centroid_distance(t2ld, dwild).cpu().numpy()
                print(rs, input_dict['t2_ldmks_paths'][i][0])
                if not np.isnan(rs):
                    results['ldmk'].append(rs)
                    results['ldmk-wo-reg'].append(rs_wo_reg)
                else:
                    print('warning: nan exists.')
                
            """save images"""
            self.save_img(fx_img, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            self.save_img(mv_img, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))
            self.save_img(warpped_img, os.path.join(visualization_path, f'{idx+1}-wp_img.nii'))

            self.save_img(input_dict['dwi_b0'].cuda(), os.path.join(visualization_path, f'{idx+1}-pr_img.nii'))
            self.save_img(warpped_pr_img, os.path.join(visualization_path, f'{idx+1}-wp_pr_img.nii'))
            # self.save_img(ddf[0, 0, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-x.nii'))
            # self.save_img(ddf[0, 1, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-y.nii'))
            # self.save_img(ddf[0, 2, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-z.nii'))
        
        """check the improvement on the worst cases"""
        case_num = 7
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




        """save results and calculate p-value of paired t-test"""
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

        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)

    @staticmethod
    def save_img(tensor_arr, save_path):
        arr = torch.squeeze(tensor_arr)
        arr = arr.cpu().numpy()
        nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
        nib_img.header['pixdim'][1:4] = np.array([1.0, 1.0, 1.0])
        nib.save(img=nib_img, filename=save_path)
        



class weakSuperVisionMpMRIReg():
    def __init__(self, config):
        self.config = config
        self.net = networks.LocalModel(input_shape=(104, 104, 92)).cuda()
        self.train_loader, self.val_loader, self.test_loader = self.set_dataloader()
        self.train_set = dataloaders.mpMRIData(config=config, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.val_set = dataloaders.mpMRIData(config=config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        self.test_set = dataloaders.mpMRIData(config=config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        self.best_metric = 0
        self.epoch = 0

    def get_log_dir(self):
        assert self.config.exp_name is not None, "exp_name should not be None."
        log_dir = os.path.join(self.config.log_path, self.config.exp_name)
        while os.path.exists(log_dir) and 'train.py' in sys.argv[0]:
            log_dir = os.path.join(self.config.log_path, self.config.exp_name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        return log_dir

    def save_configure(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'config.pkl'), 'wb') as f:
            pkl.dump(self.config, f)

    def train(self):
        self.save_configure()
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        for epoch in range(1, self.config.num_epochs + 1):
            self.epoch = epoch
            self.net.train()
            for idx, input_dict in enumerate(self.train_loader):
                print(f'epoch {epoch}, step {idx + 1}')
                if self.config.mv_mod == 'mixed':
                    fx_img, mv_img = input_dict['t2'].cuda(), input_dict['mv_img'].cuda()
                elif self.config.mv_mod == 'dwi':
                    fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()
                else:
                    raise NotImplementedError
                weakly_img = input_dict['dwi_b0'].cuda()
                optimizer.zero_grad()
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_weakly_img = warp3d(weakly_img, ddf)
                warpped_img = warp3d(mv_img, ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img, warpped_weakly_img)
                global_loss.backward()
                optimizer.step()
            if epoch % self.config.save_frequency == 0:
                self.save(epoch)
            print('-'*10, 'validation', '-'*10)
            self.validation(dataloader=self.val_loader, epoch=epoch)

    def loss(self, ddf, fx_img, wp_img, wp_wk_img):
        L_Dreg_bde = loss.bending_energy(ddf) * self.config.w_bde
        L_Dreg_l2n = loss.l2_gradient(ddf) * self.config.w_l2n
        L_Isim = (1.5 - loss.global_mutual_information(fx_img, wp_img)) * self.config.w_gmi * self.config.w_uns
        L_Isim_wk = (1.5 - loss.global_mutual_information(fx_img, wp_wk_img)) * self.config.w_gmi * self.config.w_wks
        L_All = L_Dreg_bde + L_Dreg_l2n + L_Isim + L_Isim_wk
        Info = f'L_All:{L_All:.3f}, Loss_Dreg_bde: {L_Dreg_bde:.3f}, Loss_Dreg_l2n: {L_Dreg_l2n:.3f}, Loss_Isim: {L_Isim:.3f}, Loss_Isim_wk: {L_Isim_wk:.3f}'
        print(Info)
        return L_All

    def save(self, epoch, type=None):
        ckpt_path = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)
        if type is None:
            torch.save(self.net, os.path.join(ckpt_path, f'epoch-{epoch}.pt'))
        elif type == 'best':
            exist_best_models = glob(os.path.join(ckpt_path, 'best*.pt'))
            [os.remove(i) for i in exist_best_models]
            torch.save(self.net, os.path.join(ckpt_path, f'best-epoch-{epoch}.pt'))
        else:
            pass

    @torch.no_grad()
    def validation(self, dataloader, epoch=None):
        self.net.eval()
        res = []
        for idx, input_dict in enumerate(dataloader):
            if self.config.mv_mod == 'mixed':
                fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()  # [batch, 1, x, y, z]
                weakly_img = input_dict['dwi_b0'].cuda()
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_img = warp3d(mv_img, ddf)
                warpped_weakly_img = warp3d(weakly_img, ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img, warpped_weakly_img)
                res.append(loss.global_mutual_information(fx_img, warpped_img))

                fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi_b0'].cuda()  # [batch, 1, x, y, z]
                weakly_img = input_dict['dwi_b0'].cuda()
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_img = warp3d(mv_img, ddf)
                warpped_weakly_img = warp3d(weakly_img, ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img, warpped_weakly_img)
                res.append(loss.global_mutual_information(fx_img, warpped_img))
            else:
                fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()  # [batch, 1, x, y, z]
                weakly_img = input_dict['dwi_b0'].cuda()
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_img = warp3d(mv_img, ddf)
                warpped_weakly_img = warp3d(weakly_img, ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img, warpped_weakly_img)
                res.append(loss.global_mutual_information(fx_img, warpped_img))

        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)
        if mean > self.best_metric:
            self.best_metric = mean
            self.save(epoch, 'best')
        print('Result:', mean, std)

    def load_epoch(self, num_epoch):
        if num_epoch != 'best':
            self.epoch = int(num_epoch)
            self.net = torch.load(os.path.join(self.log_dir, 'checkpoints', f'epoch-{num_epoch}.pt'))
        else:
            self.epoch = num_epoch
            best_ckpt = glob(os.path.join(self.log_dir, 'checkpoints', f'best*'))
            assert(len(best_ckpt)) != 0, "no best ckpt found in this exp..."
            self.net = torch.load(best_ckpt[0])

    def load_ckpt(self, ckpt_path):
        self.net = torch.load(ckpt_path)

    @torch.no_grad()
    def inference(self):
        self.net.eval()
        visualization_path = os.path.join(self.log_dir, f'vis-{self.epoch}')
        os.makedirs(visualization_path, exist_ok=True)
        
        results = {'mi':[], 'ldmk':[], 'mi-wo-reg':[], 'ldmk-wo-reg':[]}
        ldmk_improvement_topX = []
        ldmk_num_each_sample = []
        ldmk_footprint = []
        for idx, input_dict in enumerate(self.test_loader):
            fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()  # [batch, 1, x, y, z]
            _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
            warpped_img = warp3d(mv_img, ddf)  # [batch=1, 1, w, h, z]
            warpped_pr_img = warp3d(input_dict['dwi_b0'].cuda(), ddf)
            results['mi'].append(loss.global_mutual_information(fx_img, warpped_img).cpu().numpy())
            results['mi-wo-reg'].append(loss.global_mutual_information(fx_img, mv_img).cpu().numpy())

            t2_ldmks = input_dict['t2_ldmks'].cuda()
            dwi_ldmks = input_dict['dwi_ldmks'].cuda()

            ldmk_num_each_sample.append(t2_ldmks.shape[1])

            for i in range(t2_ldmks.shape[1]):
                ldmk_footprint.append(idx)
                t2ld = t2_ldmks[:, i:i+1, :, :, :]
                dwild = dwi_ldmks[:, i:i+1, :, :, :]
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
                    print(
                        f'{idx+1}-{i+1}',
                        t2_pid,
                        t2_ldmk_file, dwi_ldmk_file, 
                        'woreg:', np.around(rs_wo_reg, decimals=3), 
                        'after-reg:', np.around(rs, decimals=3), 
                        'ipmt:', np.around(rs_wo_reg - rs, decimals=3)
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
        #-------------------------------Table2-----------------------------
        # print('-----calculation topX----')
        # # topX misalignment by MI
        # lnes = ldmk_num_each_sample.copy()
        # pid_idx_list = list(range(len(self.test_loader)))


        # topX_num = 4
        # print(f'top {topX_num}:')
        # mi_before_reg = results['mi-wo-reg'].copy()
        # mi_after_reg = [i for _, i in sorted(zip(mi_before_reg, results['mi']))]
        # ordered_idxs = [i for _, i in sorted(zip(mi_before_reg, pid_idx_list))]
        # lnes_mi = [i for _, i in sorted(zip(mi_before_reg, lnes))]
        # mi_before_reg.sort()
        
        # mibr_mean, mibr_std = np.around(np.mean(mi_before_reg[:topX_num]), decimals=3), np.around(np.std(mi_before_reg[:topX_num]), decimals=3),
        # miar_mean, miar_std = np.around(np.mean(mi_after_reg[:topX_num]), decimals=3), np.around(np.std(mi_after_reg[:topX_num]), decimals=3)
        # print(f'largest misaglingment top {topX_num} by MI ---> bef {mibr_mean, mibr_std}, after {miar_mean, std}')
        # print('p-value:', stats.ttest_rel(mi_before_reg[:topX_num], mi_after_reg[:topX_num])[1])

        # bef_ldmks_topX, aft_ldmks_topX = [], []
        # for idx in ordered_idxs:
        #     start = sum([i for i in lnes[:idx]])
        #     end = start + lnes_mi[idx]
        #     bef_ldmks_topX.extend(results['ldmk-wo-reg'][start:end])
        #     aft_ldmks_topX.extend(results['ldmk'][start:end])
        
        # btre_mean, btre_std = np.around(np.mean(bef_ldmks_topX), decimals=3), np.around(np.std(bef_ldmks_topX), decimals=3),
        # atre_mean, atre_std = np.around(np.mean(aft_ldmks_topX), decimals=3), np.around(np.std(aft_ldmks_topX), decimals=3)
        # print(f'largest misaglingment top {topX_num} by MI (TRE)---> bef {btre_mean, btre_std}, after {atre_mean, atre_std}')
        # print('p-value:', stats.ttest_rel(bef_ldmks_topX, aft_ldmks_topX)[1])


        # MI_diff = [i - j for (i, j) in zip(results['mi'], results['mi-wo-reg'])]
        
        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)

    @staticmethod
    def save_img(tensor_arr, save_path):
        arr = torch.squeeze(tensor_arr)
        arr = arr.cpu().numpy()
        nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
        nib_img.header['pixdim'][1:4] = np.array([1.0, 1.0, 1.0])
        nib.save(img=nib_img, filename=save_path)


class joint3(object):
    def __init__(self, config):
        self.config = config
        self.net_AB = networks.LocalModel(input_shape=(104, 104, 92)).cuda()
        self.net_BC = networks.LocalModel(input_shape=(104, 104, 92)).cuda()
        self.net_AC = networks.LocalModel(input_shape=(104, 104, 92)).cuda()
        self.log_dir = self.get_log_dir()
        self.train_set = dataloaders.mpMRIData(config=config, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
        self.val_set = dataloaders.mpMRIData(config=config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
        self.test_set = dataloaders.mpMRIData(config=config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
        self.best_metric = 0
        self.epoch = 0

    def get_log_dir(self):
        assert self.config.exp_name is not None, "exp_name should not be None."
        log_dir = os.path.join(self.config.log_path, self.config.exp_name)
        while os.path.exists(log_dir) and 'train.py' in sys.argv[0]:
            log_dir = os.path.join(self.config.log_path, self.config.exp_name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S"))
        return log_dir

    def save_configure(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'config.pkl'), 'wb') as f:
            pkl.dump(self.config, f)

    def train(self):
        self.save_configure()
        # t2-dwi_b0
        self.net_AB = torch.load('./logs/08.uns-baseline-gmi0.5-bend3000-l2n3000-batch8-t2-dwi_b0/checkpoints/best-epoch-1890.pt')  
        # dwi_b0-dwi
        self.net_BC = torch.load('./logs/13.uns-baseline-gmi0.5-bend1000-l2n1000-batch8-dwi_b0-dwi/checkpoints/best-epoch-2387.pt')
        # t2-dwi
        self.net_AC = torch.load('./logs/10.uns-baseline-gmi0.5-bend3000-l2n3000-batch8-t2-dwi/checkpoints/best-epoch-1929.pt')
        

        optimizer = optim.Adam(
            list(self.net_AB.parameters())+list(self.net_BC.parameters())+list(self.net_AC.parameters()), 
            lr=self.config.lr)
        for epoch in range(1, self.config.num_epochs + 1):
            self.epoch = epoch
            self.net_AB.train()
            self.net_BC.train()
            self.net_AC.train()
            for idx, input_dict in enumerate(self.train_loader):
                print(f'epoch {epoch}, step {idx + 1}')
                t2_arr = input_dict['t2'].cuda()
                dwi_b0_arr = input_dict['dwi_b0'].cuda()
                dwi_arr = input_dict['dwi'].cuda()
                optimizer.zero_grad()

                _, ddf_AB = self.net_AB(torch.cat([t2_arr, dwi_b0_arr], dim=1))
                _, ddf_BC = self.net_BC(torch.cat([dwi_b0_arr, dwi_arr], dim=1))
                _, ddf_AC = self.net_AC(torch.cat([t2_arr, dwi_arr], dim=1))
                warpped_img_AB = warp3d(dwi_b0_arr, ddf_AB)
                warpped_img_BC = warp3d(dwi_arr, ddf_BC)
                warpped_img_AC = warp3d(dwi_arr, ddf_AC)

                mutual_info_loss = self.loss(ddf_AB, t2_arr, warpped_img_AB, prefix='Net AB:') + \
                    self.loss(ddf_BC, dwi_b0_arr, warpped_img_BC, prefix='Net BC:') + \
                    self.loss(ddf_AC, t2_arr, warpped_img_AC, prefix='Net AC:')

                warpped_img_C2A = warp3d(warp3d(dwi_arr, ddf_BC), ddf_AB)
                ddf_similarity_loss = self.config.w_dsl * loss.ssd(warpped_img_C2A, warpped_img_AC)
                global_loss = mutual_info_loss + ddf_similarity_loss
                print(f'DDF similarity: {ddf_similarity_loss:.3f}, global loss: {global_loss:.3f}')

                global_loss.backward()
                optimizer.step()
            if epoch % self.config.save_frequency == 0:
                self.save(epoch)
            print('-'*10, 'validation', '-'*10)
            self.validation(dataloader=self.val_loader, epoch=epoch)

    def loss(self, ddf, fx_img, wp_img, prefix=''):
        L_Dreg_bde = loss.bending_energy(ddf) * self.config.w_bde
        L_Dreg_l2n = loss.l2_gradient(ddf) * self.config.w_l2n
        L_Isim = (1.5 - loss.global_mutual_information(fx_img, wp_img)) * self.config.w_gmi
        L_All = L_Dreg_bde + L_Dreg_l2n + L_Isim
        Info = f'{prefix} L_All:{L_All:.3f}, Loss_Dreg_bde: {L_Dreg_bde:.3f}, Loss_Dreg_l2n: {L_Dreg_l2n:.3f}, Loss_Isim: {L_Isim:.3f}'
        print(Info)
        return L_All

    def save(self, epoch, type=None):
        ckpt_path = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)
        if type is None:
            torch.save(self.net_AB, os.path.join(ckpt_path, f'AB-epoch-{epoch}.pt'))
            torch.save(self.net_BC, os.path.join(ckpt_path, f'BC-epoch-{epoch}.pt'))
            torch.save(self.net_AC, os.path.join(ckpt_path, f'AC-epoch-{epoch}.pt'))
        elif type == 'best':
            exist_best_models = glob(os.path.join(ckpt_path, 'best*.pt'))
            [os.remove(i) for i in exist_best_models]
            torch.save(self.net_AB, os.path.join(ckpt_path, f'best-AB-epoch-{epoch}.pt'))
            torch.save(self.net_BC, os.path.join(ckpt_path, f'best-BC-epoch-{epoch}.pt'))
            torch.save(self.net_AC, os.path.join(ckpt_path, f'best-AC-epoch-{epoch}.pt'))
        else:
            pass

    @torch.no_grad()
    def validation(self, dataloader, epoch=None):
        self.net_AC.eval()
        self.net_AB.eval()
        self.net_BC.eval()
        res = []
        for idx, input_dict in enumerate(dataloader):
            t2_arr = input_dict['t2'].cuda()
            dwi_arr = input_dict['dwi'].cuda()

            _, ddf_AC = self.net_AC(torch.cat([t2_arr, dwi_arr], dim=1))
            warpped_img = warp3d(dwi_arr, ddf_AC)

            global_loss = self.loss(ddf_AC, t2_arr, warpped_img, prefix="Validation")
            res.append(loss.global_mutual_information(t2_arr, warpped_img))
        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)
        if mean > self.best_metric:
            self.best_metric = mean
            self.save(epoch, 'best')
        print('Result:', mean, std)

    def load_epoch(self, num_epoch):
        if num_epoch != 'best':
            self.epoch = int(num_epoch)
            self.net_AB = torch.load(os.path.join(self.log_dir, 'checkpoints', f'AB-epoch-{num_epoch}.pt'))
            self.net_BC = torch.load(os.path.join(self.log_dir, 'checkpoints', f'BC-epoch-{num_epoch}.pt'))
            self.net_AC = torch.load(os.path.join(self.log_dir, 'checkpoints', f'AC-epoch-{num_epoch}.pt'))
        else:
            self.epoch = num_epoch
            best_ckpt = glob(os.path.join(self.log_dir, 'checkpoints', f'best*'))
            assert(len(best_ckpt)) != 0, "no best ckpt found in this exp..."
            self.net_AB = torch.load([i for i in best_ckpt if 'AB-' in i][0])
            self.net_BC = torch.load([i for i in best_ckpt if 'BC-' in i][0])
            self.net_AC = torch.load([i for i in best_ckpt if 'AC-' in i][0])

    @torch.no_grad()
    def inference(self):
        """need to modify"""
        self.net_AC.eval()
        self.net_AB.eval()
        self.net_BC.eval()
        visualization_path = os.path.join(self.log_dir, f'vis-{self.epoch}')
        os.makedirs(visualization_path, exist_ok=True)
        
        results = {'mi':[], 'ldmk':[], 'mi-wo-reg':[], 'ldmk-wo-reg':[]}
        for idx, input_dict in enumerate(self.test_loader):
            t2_arr = input_dict['t2'].cuda()
            dwi_arr = input_dict['dwi'].cuda()

            _, ddf_AC = self.net_AC(torch.cat([dwi_arr, t2_arr], dim=1))
            warpped_img = warp3d(dwi_arr, ddf_AC)  # [batch=1, 1, w, h, z]
            warpped_pr_img = warp3d(input_dict['dwi_b0'].cuda(), ddf_AC)

            results['mi'].append(loss.global_mutual_information(t2_arr, warpped_img).cpu().numpy())
            results['mi-wo-reg'].append(loss.global_mutual_information(t2_arr, dwi_arr).cpu().numpy())
            t2_ldmks = input_dict['t2_ldmks'].cuda()
            dwi_ldmks = input_dict['dwi_ldmks'].cuda()

            """calculating landmark distances"""
            for i in range(t2_ldmks.shape[1]):
                t2ld = t2_ldmks[:, i:i+1, :, :, :]
                dwild = dwi_ldmks[:, i:i+1, :, :, :]
                wpld = warp3d(dwild, ddf_AC)
                rs = loss.centroid_distance(t2ld, wpld).cpu().numpy()
                rs_wo_reg = loss.centroid_distance(t2ld, dwild).cpu().numpy()
                print(rs)
                if not np.isnan(rs):
                    results['ldmk'].append(rs)
                    results['ldmk-wo-reg'].append(rs_wo_reg)
                else:
                    print('warning: nan exists.')

            self.save_img(t2_arr, os.path.join(visualization_path, f'{idx+1}-fx_img.nii'))
            self.save_img(dwi_arr, os.path.join(visualization_path, f'{idx+1}-mv_img.nii'))
            self.save_img(warpped_img, os.path.join(visualization_path, f'{idx+1}-wp_img.nii'))

            self.save_img(input_dict['dwi_b0'].cuda(), os.path.join(visualization_path, f'{idx+1}-pr_img.nii'))
            self.save_img(warpped_pr_img, os.path.join(visualization_path, f'{idx+1}-wp_pr_img.nii'))
            # self.save_img(ddf_AC[0, 0, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-x.nii'))
            # self.save_img(ddf_AC[0, 1, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-y.nii'))
            # self.save_img(ddf_AC[0, 2, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-z.nii'))

        """save results and calculate p-value of paired t-test"""
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

        with open(os.path.join(self.log_dir, 'results.pkl'), 'wb') as f:
            pkl.dump(results, f)

    @staticmethod
    def save_img(tensor_arr, save_path):
        arr = torch.squeeze(tensor_arr)
        arr = arr.cpu().numpy()
        nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
        nib_img.header['pixdim'][1:4] = np.array([1.0, 1.0, 1.0])
        nib.save(img=nib_img, filename=save_path)







class test_affine_optim(object):
    def __init__(self, config):
        self.config = config
        self.train_set = dataloaders.mpMRIData(config=config, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=1, shuffle=False)
        self.epoch = 0
        with open('./do_unfinished_cv.pkl', 'rb') as f:
            self.unfinished = pkl.load(f)[f'cv{self.config.cv}']

    @staticmethod
    def save_img(tensor_arr, save_path, imtype='nifti'):
        save_folder = os.path.dirname(save_path)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        arr = torch.squeeze(tensor_arr)
        assert len(arr.shape)==3, "not a 3 dimentional volume, need to check."
        arr = arr.cpu().numpy()

        if imtype=='nifty':
            nib_img = nib.Nifti1Image(arr, affine=np.eye(4))
            nib_img.header['pixdim'][1:4] = np.array([1.0, 1.0, 1.0])
            if (not save_path.endswith('.nii')) and (not save_path.endswith('.nii.gz')):
                save_path += '.nii.gz'
            nib.save(img=nib_img, filename=save_path)
        elif imtype=='numpy':
            if not save_path.endswith('.npy'):
                save_path += '.npy'
            np.save(save_path, arr)
        else:
            raise NotImplementedError


    def train(self):
        affine_transform = tio.RandomAffine()
        # affine_transform = RandomElasticDeformation(num_control_points=(7, 7, 7), locked_borders=2)

        for step, input_dict in enumerate(self.train_loader):
            if input_dict['dwi_b0_path'][0].split('/')[-2] not in self.unfinished:
                continue

            b0_arr = input_dict['dwi_b0'].cuda()
            hb_arr = input_dict['dwi'].cuda()
            t2_arr = input_dict['t2']
            best_mi = ori_mi = loss.global_mutual_information(b0_arr, hb_arr)
            save_path = os.path.dirname(input_dict['dwi_b0_path'][0])

            print(f'CV {self.config.cv} | image {step + 1 - 30*self.config.cv} / {30} : {save_path}')

            for idx in tqdm(range(self.config.repeat_num)):
                new_b0 = affine_transform(b0_arr[0].cpu())
                new_b0 = torch.unsqueeze(new_b0, dim=0).cuda()
                mi = loss.global_mutual_information(new_b0, hb_arr)
                if mi > best_mi:
                    best_mi = mi
                    self.save_img(new_b0, os.path.join(save_path, 'dwi_b0_afTrans'), 'numpy')
                    self.save_img(new_b0, os.path.join(save_path.replace('52-52-46', '52-52-46-nii'), 'dwi_b0_afTrans'), 'nifty')
                    self.save_img(b0_arr, os.path.join(save_path.replace('52-52-46', '52-52-46-nii'), 'dwi_b0'), 'nifty')
                    self.save_img(hb_arr, os.path.join(save_path.replace('52-52-46', '52-52-46-nii'), 'dwi'), 'nifty')
                    self.save_img(t2_arr, os.path.join(save_path.replace('52-52-46', '52-52-46-nii'), 't2'), 'nifty')

            if best_mi > ori_mi:
                print(f'higher MI get, ori:{ori_mi}, best:{best_mi}')

