import src.model.networks as networks
from src.model import loss
from src.model.functions import warp3d,ddf_merge
from src.model.archs.baseArch import BaseArch
from src.data import dataloaders
import torch, os
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np
from scipy import stats


class mpMRIRegUnsupervise(object):
    def __init__(self, config):
        self.net = networks.LocalModel(input_shape=(104, 104, 92)).cuda()
        self.train_loader, self.val_loader, self.test_loader = self.set_dataloader()
        self.best_metric = 0

    def set_dataloader(self):
        self.train_set = dataloaders.mpMRIData(config=self.config, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
        print('>>> Train set ready.')
        self.val_set = dataloaders.mpMRIData(config=self.config, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        print('>>> Validation set ready.')
        self.test_set = dataloaders.mpMRIData(config=self.config, phase='test')
        self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        print('>>> Holdout set ready.')
        return train_loader, val_loader, test_loader

    def get_input(self, input_dict):
        if self.config.mv_mod in ['dwi', 'dwi_b0']:
            fx_img, mv_img = input_dict[self.config.fx_mod].cuda(), input_dict[self.config.mv_mod].cuda()  # [batch, 1, x, y, z]
        elif self.config.mv_mod == 'mixed':
            fx_img, mv_img = input_dict[self.config.fx_mod].cuda(), input_dict['mv_img'].cuda()  # [batch, 1, x, y, z]
        else:
            raise NotImplementedError
        return fx_img, mv_img 

    def train(self):
        self.save_configure()
        optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        for self.epoch in range(1, self.config.num_epochs + 1):
            self.net.train()
            for self.step, input_dict in enumerate(self.train_loader):
                fx_img, mv_img = self.get_input(input_dict)
                optimizer.zero_grad()
                _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
                warpped_img = warp3d(mv_img, ddf)
                global_loss = self.loss(ddf, fx_img, warpped_img)
                global_loss.backward()
                optimizer.step()
            if self.epoch % self.config.save_frequency == 0:
                self.save(self.epoch)
            print('-'*10, 'validation', '-'*10)
            self.validation(dataloader=self.val_loader, epoch=self.epoch)

    def loss(self, ddf, fx_img, wp_img):
        L_Dreg_bde = loss.bending_energy(ddf) * self.config.w_bde
        L_Dreg_l2n = loss.l2_gradient(ddf) * self.config.w_l2n
        L_Isim = (1.5 - loss.global_mutual_information(fx_img, wp_img)) * self.config.w_gmi
        L_All = L_Dreg_bde + L_Dreg_l2n + L_Isim
        Info = f'epoch {self.epoch}, step {self.step+1}, L_All:{L_All:.3f}, Loss_Dreg_bde: {L_Dreg_bde:.3f}, Loss_Dreg_l2n: {L_Dreg_l2n:.3f}, Loss_Isim: {L_Isim:.3f}'
        print(Info)
        return L_All

    @torch.no_grad()
    def validation(self, dataloader, epoch=None):
        # self.net.eval()
        res = []
        for idx, input_dict in enumerate(dataloader):
            fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()  # [batch, 1, x, y, z]
            _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
            warpped_img = warp3d(mv_img, ddf)
            global_loss = self.loss(ddf, fx_img, warpped_img)
            res.append(loss.global_mutual_information(fx_img, warpped_img))
        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)
        if mean > self.best_metric:
            self.best_metric = mean
            self.save(epoch, 'best')
        print('Result:', mean, std)


    @torch.no_grad()
    def evaluation(self):
        """only test on fx and mv pairs"""
        # self.net.eval()
        visualization_path = os.path.join(self.log_dir, f'vis-{self.epoch}')
        os.makedirs(visualization_path, exist_ok=True)
        
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
            results['mi'].append(loss.global_mutual_information(fx_img, warpped_img).cpu().numpy())
            results['mi-wo-reg'].append(loss.global_mutual_information(fx_img, mv_img).cpu().numpy())

            t2_ldmks = input_dict['t2_ldmks'].cuda()
            dwi_ldmks = input_dict['dwi_ldmks'].cuda()

            """calculating landmark distances"""
            for i in range(t2_ldmks.shape[1]):
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
            self.save_img(ddf[0, 0, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-x.nii'))
            self.save_img(ddf[0, 1, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-y.nii'))
            self.save_img(ddf[0, 2, :, :, :], os.path.join(visualization_path, f'{idx+1}-ddf-z.nii'))
        
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