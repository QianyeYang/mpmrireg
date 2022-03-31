from src.model.networks.local import LocalModel
from src.model import loss
import src.model.functions as smfunctions
from src.model.archs.baseArch import BaseArch
from src.data import dataloaders
import torch, os
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle as pkl
import numpy as np
from scipy import stats
from glob import glob


class mpmrireg(BaseArch):
    def __init__(self, config):
        super(mpmrireg, self).__init__(config)
        self.config = config
        self.check_methods()
        self.set_dataloader()
        self.set_networks_and_optim()
        self.best_metric = 0

    def check_methods(self):
        assert self.config.method in ['unsupervised', 'mixed', 'B0', 'joint', 'privileged'], f"method {self.config.method} can not be recongnised."

    def set_networks_and_optim(self):
        if self.config.method != "joint":
            self.net = LocalModel(self.config).cuda()
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr)
        else:
            self.net_AB = LocalModel(self.config).cuda()
            self.net_BC = LocalModel(self.config).cuda()
            self.net_AC = LocalModel(self.config).cuda()
            self.optimizer = optim.Adam(
                list(self.net_AB.parameters())+list(self.net_BC.parameters())+list(self.net_AC.parameters()), 
                lr=self.config.lr
                )

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

    def set_external_dataloader(self):
        self.miami_set = dataloaders.mpMRIData(config=self.config, phase='test', external='/media/yipeng/data/data/mpMriReg/external/Miami-external-npy')
        self.miami_loader = DataLoader(self.miami_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        self.cia_set = dataloaders.mpMRIData(config=self.config, phase='test', external='/media/yipeng/data/data/mpMriReg/external/CIA-external-npy')
        self.cia_loader = DataLoader(self.cia_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        print('>>> External set ready.')
        

    def get_input(self, input_dict, phase='train'):
        assert phase in ['train', 'val', 'test'], f"phase {phase} not correct for getting data"
        if self.config.method == 'unsupervised':
            data = self.get_input_unsup(input_dict, phase)
        elif self.config.method == 'mixed':
            data = self.get_input_mixed(input_dict, phase)
        elif self.config.method == 'joint':
            data = self.get_input_joint(input_dict, phase)
        elif self.config.method == 'privileged':
            data = self.get_input_privileged(input_dict, phase)
        elif self.config.method == 'B0':
            data = self.get_input_B0(input_dict, phase)
        return data 

    def get_input_unsup(self, input_dict, phase):
        fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()
        if phase == 'train':
            affine_grid_fx = smfunctions.rand_affine_grid(fx_img, scale=self.config.affine_scale)
            affine_grid_mv = smfunctions.rand_affine_grid(mv_img, scale=self.config.affine_scale)
            mv_img = torch.nn.functional.grid_sample(mv_img, affine_grid_mv, mode='bilinear', align_corners=True)
            fx_img = torch.nn.functional.grid_sample(fx_img, affine_grid_fx, mode='bilinear', align_corners=True)

        return fx_img, mv_img
        
    def get_input_mixed(self, input_dict, phase):
        fx_img, mv_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda()
        if phase == 'train':
            affine_grid_fx = smfunctions.rand_affine_grid(fx_img, scale=self.config.affine_scale)
            affine_grid_mv = smfunctions.rand_affine_grid(mv_img, scale=self.config.affine_scale)
            mv_img = torch.nn.functional.grid_sample(mv_img, affine_grid_mv, mode='bilinear', align_corners=True)
            fx_img = torch.nn.functional.grid_sample(fx_img, affine_grid_fx, mode='bilinear', align_corners=True)
        
        return fx_img, mv_img 

    def get_input_B0(self, input_dict, phase):
        fx_img = input_dict['t2'].cuda()
        mv_img = input_dict['dwi_b0'].cuda()

        if phase == 'train':
            affine_grid_fx = smfunctions.rand_affine_grid(fx_img, scale=self.config.affine_scale)
            affine_grid_mv = smfunctions.rand_affine_grid(mv_img, scale=self.config.affine_scale)
            mv_img = torch.nn.functional.grid_sample(mv_img, affine_grid_mv, mode='bilinear', align_corners=True)
            fx_img = torch.nn.functional.grid_sample(fx_img, affine_grid_fx, mode='bilinear', align_corners=True)

        return fx_img, mv_img  

    def get_input_joint(self, input_dict, phase):
        t2, dwi, dwi_b0 = input_dict['t2'].cuda(), input_dict['dwi'].cuda(), input_dict['dwi_b0'].cuda()
        if phase == 'train':
            affine_grid_t2 = smfunctions.rand_affine_grid(t2, scale=self.config.affine_scale)
            affine_grid_dwis = smfunctions.rand_affine_grid(dwi, scale=self.config.affine_scale)
            t2 = torch.nn.functional.grid_sample(t2, affine_grid_t2, mode='bilinear', align_corners=True)
            dwi = torch.nn.functional.grid_sample(dwi, affine_grid_dwis, mode='bilinear', align_corners=True)
            dwi_b0 = torch.nn.functional.grid_sample(dwi_b0, affine_grid_dwis, mode='bilinear', align_corners=True)

        return t2, dwi, dwi_b0
        

    def get_input_privileged(self, input_dict, phase):
        t2, dwi, dwi_b0 = input_dict['t2'].cuda(), input_dict['dwi'].cuda(), input_dict['dwi_b0'].cuda()

        if phase == 'train':  ## overall-level affine transformation augmentation
            affine_grid_t2 = smfunctions.rand_affine_grid(t2, scale=self.config.affine_scale)
            affine_grid_dwis = smfunctions.rand_affine_grid(dwi, scale=self.config.affine_scale)
            t2 = torch.nn.functional.grid_sample(t2, affine_grid_t2, mode='bilinear', align_corners=True)
            dwi = torch.nn.functional.grid_sample(dwi, affine_grid_dwis, mode='bilinear', align_corners=True)
            dwi_b0 = torch.nn.functional.grid_sample(dwi_b0, affine_grid_dwis, mode='bilinear', align_corners=True)

        if phase == 'train': ## privileged specific affine optimization(correction)
            base_mi = [loss.global_mutual_information(dwi[i], dwi_b0[i]) for i in range(dwi.shape[0])]
            for i in range(self.config.mi_resample_count):
                affine_grid_b0 = smfunctions.rand_affine_grid(dwi_b0, scale=self.config.affine_scale)
                new_b0 = torch.nn.functional.grid_sample(dwi_b0, affine_grid_dwis, mode='bilinear', align_corners=True)
                affine_mi = [loss.global_mutual_information(dwi[i], new_b0[i]) for i in range(dwi.shape[0])]

                for idx, (b_mi, a_mi) in enumerate(zip(base_mi, affine_mi)):
                    if a_mi > b_mi:
                        print("better mi found!!!")
                        dwi_b0[idx] = new_b0[idx]
                        base_mi[idx] = a_mi

        return t2, dwi, dwi_b0

    def set_train_mode(self):
        if self.config.method != "joint":
            self.net.train()
        else:
            self.net_AB.train()
            self.net_BC.train()
            self.net_AC.train()

    def set_eval_mode(self):
        if self.config.method != "joint":
            self.net.eval()
        else:
            self.net_AB.eval()
            self.net_BC.eval()
            self.net_AC.eval()

    def forward(self, input_data):
        if self.config.method == "joint":
            t2, dwi, dwi_b0 = input_data
            _, ddf_AB = self.net_AB(torch.cat([t2, dwi_b0], dim=1))
            _, ddf_BC = self.net_BC(torch.cat([dwi_b0, dwi], dim=1))
            _, ddf_AC = self.net_AC(torch.cat([t2, dwi], dim=1))
            return ddf_AB, ddf_BC, ddf_AC
        elif self.config.method == "privileged":
            t2, dwi, _ = input_data
            _, ddf = self.net(torch.cat([t2, dwi], dim=1))
            return ddf
        else:
            _, ddf = self.net(torch.cat(input_data, dim=1))
            return ddf

    def get_warpped_images(self, input_data, ddfs):
        if self.config.method == "joint":
            t2, dwi, dwi_b0 = input_data
            ddf_AB, ddf_BC, ddf_AC = ddfs
            warpped_img_AB = smfunctions.warp3d(dwi_b0, ddf_AB)
            warpped_img_BC = smfunctions.warp3d(dwi, ddf_BC)
            warpped_img_AC = smfunctions.warp3d(dwi, ddf_AC)
            warpped_img_C2A = smfunctions.warp3d(smfunctions.warp3d(dwi, ddf_BC), ddf_AB)
            warpped_imgs = [warpped_img_AB, warpped_img_AC, warpped_img_BC, warpped_img_C2A]
        elif self.config.method == "privileged":
            _, mv_img, b0 = input_data  # t2, dwi, dwi_b0
            warpped_imgs = [smfunctions.warp3d(mv_img, ddfs), smfunctions.warp3d(b0, ddfs)]
        else:
            _, mv_img = input_data  # t2, dwi
            warpped_imgs = smfunctions.warp3d(mv_img, ddfs)
        return warpped_imgs

    def train(self):
        self.save_configure()

        for self.epoch in range(1, self.config.num_epochs + 1):
            self.set_train_mode()
            print('-'*10, 'training', '-'*10)

            for self.step, input_dict in enumerate(self.train_loader):
                input_data = self.get_input(input_dict)
                self.optimizer.zero_grad()
                ddfs = self.forward(input_data)
                warpped_imgs = self.get_warpped_images(input_data, ddfs)

                global_loss = self.loss(input_data, ddfs, warpped_imgs)
                global_loss.backward()
                self.optimizer.step()

            if self.epoch % self.config.save_frequency == 0:
                self.SAVE()
            print('-'*10, 'validation', '-'*10)
            self.validation(dataloader=self.val_loader, epoch=self.epoch)

    def SAVE(self, type=None):
        if self.config.method != "joint":
            self.save(type)
        else:
            ckpt_path = os.path.join(self.log_dir, 'checkpoints')
            os.makedirs(ckpt_path, exist_ok=True)
            if type is None:
                torch.save(self.net_AB, os.path.join(ckpt_path, f'AB-epoch-{self.epoch}.pt'))
                torch.save(self.net_BC, os.path.join(ckpt_path, f'BC-epoch-{self.epoch}.pt'))
                torch.save(self.net_AC, os.path.join(ckpt_path, f'AC-epoch-{self.epoch}.pt'))
            elif type == 'best':
                exist_best_models = glob(os.path.join(ckpt_path, 'best*.pt'))
                [os.remove(i) for i in exist_best_models]
                torch.save(self.net_AB, os.path.join(ckpt_path, f'best-AB-epoch-{self.epoch}.pt'))
                torch.save(self.net_BC, os.path.join(ckpt_path, f'best-BC-epoch-{self.epoch}.pt'))
                torch.save(self.net_AC, os.path.join(ckpt_path, f'best-AC-epoch-{self.epoch}.pt'))
            else:
                raise NotImplementedError

    def load_epoch_origin(self, num_epoch):
        if num_epoch != 'best':
            self.epoch = int(num_epoch)
            self.net = torch.load(os.path.join(self.log_dir, 'checkpoints', f'epoch-{num_epoch}.pt'))
            print(f'load from epoch {self.epoch}')
        else:
            best_ckpt = glob(os.path.join(self.log_dir, 'checkpoints', 'best*'))
            assert(len(best_ckpt)) != 0, "no best ckpt found in this exp..."
            self.net = torch.load(best_ckpt[0])
            self.epoch = int(best_ckpt[0].replace('.pt', '').split('-')[-1])
            print(f'load from best epoch {best_ckpt[0]}')
    
    def load_epoch_joint(self, num_epoch):
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
    
    def load_epoch(self, num_epoch):
        if self.config.method != 'joint':
            self.load_epoch_origin(num_epoch)
        else:
            self.load_epoch_joint(num_epoch)

    def regular_loss(self, input_data, ddfs, warpped_imgs, prefix=''):
        fx_img = input_data[0]
        L_Dreg_l2g = loss.l2_gradient(ddfs) * self.config.w_l2g
        L_Isim = (1.5 - loss.global_mutual_information(fx_img, warpped_imgs)) * self.config.w_gmi

        L_All = L_Dreg_l2g + L_Isim
        Info = f'epoch {self.epoch}, step {self.step+1}, L_All:{L_All:.3f}, L2R: {L_Dreg_l2g:.6f}, Isim: {L_Isim:.3f}'
        
        print(prefix, Info)
        return L_All

    def joint_loss(self, input_data, ddfs, warpped_imgs):
        t2, dwi, dwi_b0 = input_data
        warpped_img_AB, warpped_img_AC, warpped_img_BC, warpped_img_C2A = warpped_imgs
        ddf_AB, ddf_BC, ddf_AC = ddfs
        
        l1 = self.regular_loss([t2], ddf_AB, warpped_img_AB, prefix='Net AB:')
        l2 = self.regular_loss([dwi_b0], ddf_BC, warpped_img_BC, prefix='Net BC:')
        l3 = self.regular_loss([t2], ddf_AC, warpped_img_AC, prefix='Net AC:')

        ddf_similarity_loss = self.config.w_dsl * loss.ssd(warpped_img_C2A, warpped_img_AC)
        L_All = l1 + l2 + l3 + ddf_similarity_loss
        print(f'DDF similarity: {ddf_similarity_loss:.6f}, L_All: {L_All:.3f}')
        return L_All

    def privileged_loss(self, input_data, ddfs, warpped_imgs, prefix=''):
        fx_img = input_data[0]
        L_Dreg_l2g = loss.l2_gradient(ddfs) * self.config.w_l2g
        L_Isim = (1.5 - loss.global_mutual_information(fx_img, warpped_imgs[0])) * self.config.w_gmi
        L_Isim_weak = (1.5 - loss.global_mutual_information(fx_img, warpped_imgs[1])) * self.config.w_gmi

        L_All = L_Dreg_l2g + L_Isim + L_Isim_weak
        Info = f'epoch {self.epoch}, step {self.step+1}, L_All:{L_All:.3f}, L2R: {L_Dreg_l2g:.6f}, Isim: {L_Isim:.3f}, Isim_weak: {L_Isim_weak:.3f}'
        
        print(prefix, Info)
        return L_All
        
    def loss(self, input_data, ddfs, warpped_imgs):
        if self.config.method == 'joint':
            return self.joint_loss(input_data, ddfs, warpped_imgs)
        elif self.config.method == 'privileged':
            return self.privileged_loss(input_data, ddfs, warpped_imgs)
        else:
            return self.regular_loss(input_data, ddfs, warpped_imgs)

    @torch.no_grad()
    def validation(self, dataloader, epoch=None):
        self.set_eval_mode()
        res = []
        for idx, input_dict in enumerate(dataloader):
            input_data = self.get_input(input_dict, phase='val')
            ddfs = self.forward(input_data)
            warpped_imgs = self.get_warpped_images(input_data, ddfs)

            fx_img = input_data[0]
            if self.config.method == 'joint':
                wp_img = warpped_imgs[1]
            elif self.config.method == 'privileged':
                wp_img = warpped_imgs[0]
            else:
                wp_img = warpped_imgs
            res.append(loss.global_mutual_information(fx_img, wp_img))
        
        res = torch.tensor(res)
        mean, std = torch.mean(res), torch.std(res)
        if mean > self.best_metric:
            self.best_metric = mean
            self.SAVE(type='best')
        print('Result:', mean, std)

    def visualization(
        self, idx, 
        fx_img, mv_img, pr_img, wp_mv_img, wp_pr_img, 
        ddf, 
        t2_ldmk_paths, dwi_ldmk_paths, t2_ldmks, dwi_ldmks, warped_ldmks, save_suffix=''):
        # save images: t2, dwi, b0, wp_dwi, wp_b0 and ddfs 

        visualization_path = os.path.join(self.log_dir, f'vis-{self.epoch}{save_suffix}', idx)
        os.makedirs(visualization_path, exist_ok=True)

        print('-' * 20)
        self.save_img(fx_img, os.path.join(visualization_path, f'fx_img.nii'))  # t2
        self.save_img(mv_img, os.path.join(visualization_path, f'mv_img.nii'))  # dwi 
        self.save_img(pr_img, os.path.join(visualization_path, f'pr_img.nii'))  # b0
        self.save_img(wp_mv_img, os.path.join(visualization_path, f'wp_mv_img.nii'))  # wp_dwi
        self.save_img(wp_pr_img, os.path.join(visualization_path, f'wp_pr_img.nii'))  # wp_b0

        self.save_img(ddf[0, 0, :, :, :], os.path.join(visualization_path, f'ddf-x.nii'))
        self.save_img(ddf[0, 1, :, :, :], os.path.join(visualization_path, f'ddf-y.nii'))
        self.save_img(ddf[0, 2, :, :, :], os.path.join(visualization_path, f'ddf-z.nii'))

        assert len(t2_ldmk_paths)==len(dwi_ldmk_paths)==len(t2_ldmks)==len(dwi_ldmks)==len(warped_ldmks), "landmark size not equal"
        # print("ldmk_list", t2_ldmk_paths)
        for i in range(len(t2_ldmks)):
            self.save_img(t2_ldmks[i], os.path.join(visualization_path, os.path.basename(t2_ldmk_paths[i][0]).replace('.npy', '_fx.nii')))
            self.save_img(dwi_ldmks[i], os.path.join(visualization_path, os.path.basename(dwi_ldmk_paths[i][0]).replace('.npy', '_mv.nii')))
            self.save_img(warped_ldmks[i], os.path.join(visualization_path, os.path.basename(dwi_ldmk_paths[i][0]).replace('.npy', '_wp.nii')))

    @torch.no_grad()
    def evaluation(self, fx_img, mv_img, pr_img):
        _, ddf = self.net(torch.cat([fx_img, mv_img], dim=1))
        wp_mv_img = smfunctions.warp3d(mv_img, ddf)
        wp_pr_img = smfunctions.warp3d(pr_img, ddf)
        return wp_mv_img, wp_pr_img, ddf

    @torch.no_grad()
    def evaluation_joint(self, fx_img, mv_img, pr_img):
        t2, dwi, dwi_b0 = fx_img, mv_img, pr_img
        _, ddf_AB = self.net_AB(torch.cat([t2, dwi_b0], dim=1))
        _, ddf_AC = self.net_AC(torch.cat([t2, dwi], dim=1))
        wp_pr_img = smfunctions.warp3d(dwi_b0, ddf_AB)
        wp_mv_img = smfunctions.warp3d(dwi, ddf_AC)
        return wp_mv_img, wp_pr_img, ddf_AC

    @torch.no_grad()
    def evaluation_B0(self, fx_img, mv_img, pr_img):
        t2, dwi, dwi_b0 = fx_img, mv_img, pr_img
        _, ddf = self.net(torch.cat([fx_img, pr_img], dim=1))
        wp_mv_img = smfunctions.warp3d(mv_img, ddf)
        wp_pr_img = smfunctions.warp3d(pr_img, ddf)
        return wp_mv_img, wp_pr_img, ddf

    @torch.no_grad()
    def calc_TREs(self, t2_ldmks, t2_ldmk_paths, dwi_ldmks, dwi_ldmk_paths, ddf):
        assert t2_ldmks.shape == dwi_ldmks.shape, "shape of the landmarks are not equal"
        assert len(t2_ldmk_paths) == len(dwi_ldmk_paths), "lens of the landmarks not equal" 
        
        temp_dict = {'tre':[], 'tre-wo-reg':[]}
        tre_dict = {}

        print('-'*10 + t2_ldmk_paths[0][0].split('/')[-2] + '-'*10)

        fx_ldmks_arr = []
        mv_ldmks_arr = []
        wp_ldmks_arr = []

        pid = os.path.basename(os.path.dirname(t2_ldmk_paths[0][0]))
        tre_dict[pid] = []

        for i in range(t2_ldmks.shape[1]):
            mv_ldmk = dwi_ldmks[:, i:i+1, :, :, :]
            fx_ldmk = t2_ldmks[:, i:i+1, :, :, :]
            wp_ldmk = smfunctions.warp3d(mv_ldmk, ddf)

            fx_ldmks_arr.append(fx_ldmk)
            mv_ldmks_arr.append(mv_ldmk)
            wp_ldmks_arr.append(wp_ldmk)

            TRE = loss.centroid_distance(fx_ldmk, wp_ldmk).cpu().numpy()
            TRE_wo_reg = loss.centroid_distance(fx_ldmk, mv_ldmk).cpu().numpy()
                
            if not np.isnan(TRE):
                temp_dict['tre'].append(TRE)
                temp_dict['tre-wo-reg'].append(TRE_wo_reg)
                tre_dict[pid].append([TRE_wo_reg, TRE])

                print(
                    f'{i+1}',
                    'woreg:', np.around(TRE_wo_reg, decimals=3),
                    'after-reg:', np.around(TRE, decimals=3),
                    'ipmt:', np.around(TRE_wo_reg - TRE, decimals=3)
                )
            else:
                print(i+1, 'warning: nan exists.')
        return temp_dict, fx_ldmks_arr, mv_ldmks_arr, wp_ldmks_arr, tre_dict
        

    def inference(self):
        self.sub_inference(external_dataloader=self.test_loader)

        self.set_external_dataloader()
        print('-------Miami results-------')
        self.sub_inference(external_dataloader=self.miami_loader, save_suffix='_miami')
        print('-------CIA results-------')
        self.sub_inference(external_dataloader=self.cia_loader, save_suffix='_cia')
        
    @torch.no_grad()
    def sub_inference(self, external_dataloader, save_suffix=''):

        self.set_eval_mode()
        results = {
            'mi': [],
            'mi-wo-reg': [],
            'tre': [],
            'tre-wo-reg': []
        }
        tre_dict = {}

        dataloader = external_dataloader

        for idx, input_dict in enumerate(dataloader):
            fx_img, mv_img, pr_img = input_dict['t2'].cuda(), input_dict['dwi'].cuda(), input_dict['dwi_b0'].cuda()
            t2_ldmks, dwi_ldmks = input_dict['t2_ldmks'].cuda(), input_dict['dwi_ldmks'].cuda()
            t2_ldmk_paths, dwi_ldmk_paths = input_dict['t2_ldmks_paths'], input_dict['dwi_ldmks_paths']

            # get ddfs, get warpped images 
            if self.config.method in ['unsupervised', 'mixed', 'privileged']:
                wp_mv_img, wp_pr_img, ddf = self.evaluation(fx_img, mv_img, pr_img)
            elif self.config.method == 'joint':
                wp_mv_img, wp_pr_img, ddf = self.evaluation_joint(fx_img, mv_img, pr_img)
            elif self.config.method == 'B0':
                wp_mv_img, wp_pr_img, ddf = self.evaluation_B0(fx_img, mv_img, pr_img)
            else:
                print("can not recognise method")
                raise NotImplementedError

            # calculate TREs and MIs 
            tmp_TREs, fx_ldmks, mv_ldmks, warped_ldmks, sub_tre_dict = self.calc_TREs(t2_ldmks, t2_ldmk_paths, dwi_ldmks, dwi_ldmk_paths, ddf)
            for key, value in tmp_TREs.items():
                results[key].extend(value)
            results['mi'].append(loss.global_mutual_information(fx_img, wp_mv_img).cpu().numpy())
            results['mi-wo-reg'].append(loss.global_mutual_information(fx_img, mv_img).cpu().numpy())
            tre_dict.update(sub_tre_dict)

            # calculate jaccobian
            jc = loss.jacobian_determinant(ddf)
            if (jc>0).all():
                print('jaccobian all > 0')
            else:
                print('jaccobian <=0 exist.')
            
            # save images  ??
            self.visualization(
                t2_ldmk_paths[0][0].split('/')[-2], 
                fx_img, 
                mv_img, 
                pr_img, 
                wp_mv_img, 
                wp_pr_img, 
                ddf,
                t2_ldmk_paths,
                dwi_ldmk_paths,
                fx_ldmks,
                mv_ldmks,
                warped_ldmks,
                save_suffix)

        # save results
        for k, v in results.items():
            print(k, np.mean(v), np.std(v))

        with open(os.path.join(self.log_dir, f'results{save_suffix}.pkl'), 'wb') as f:
            pkl.dump(results, f)

        with open(os.path.join(self.log_dir, f'tre_dic{save_suffix}.pkl'), 'wb') as f:
            pkl.dump(tre_dict, f)
        