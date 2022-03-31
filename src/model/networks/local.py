import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import src.model.layers as layers


class LocalModel(nn.Module):
    def __init__(self, config):
        super(LocalModel, self).__init__()
        self.input_shape = config.input_shape if config.patched == 0 else config.patch_size
        self.ddf_levels = config.ddf_levels

        nc = [config.nc_initial*(2**i) for i in range(5)]
        up_sz = self.calc_upsample_layer_output_size(self.input_shape, 4)
        # print('up_sz', up_sz)

        self.downsample_block0 = layers.DownsampleBlock(inc=config.inc, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.conv_block = layers.Conv3dBlock(inc=nc[3], outc=nc[4])

        self.upsample_block0 = layers.UpSampleBlock(inc=nc[4], outc=nc[3], output_size=up_sz[0])
        self.upsample_block1 = layers.UpSampleBlock(inc=nc[3], outc=nc[2], output_size=up_sz[1])
        self.upsample_block2 = layers.UpSampleBlock(inc=nc[2], outc=nc[1], output_size=up_sz[2])
        self.upsample_block3 = layers.UpSampleBlock(inc=nc[1], outc=nc[0], output_size=up_sz[3])
        self.ddf_fuse_layers = [layers.DDFFusion(inc=nc[4 - i], out_shape=self.input_shape).cuda() for i in self.ddf_levels]
    

    def calc_upsample_layer_output_size(self, input_shape, num_downsample_layers=4):
        shape = np.array(input_shape)
        tmp = [list(shape//(2**i)) for i in range(num_downsample_layers)]
        tmp.reverse()

        return tmp
        
    def forward(self, x):
        f_down0, f_jc0 = self.downsample_block0(x)
        f_down1, f_jc1 = self.downsample_block1(f_down0)
        f_down2, f_jc2 = self.downsample_block2(f_down1)
        f_down3, f_jc3 = self.downsample_block3(f_down2)
        f_bottleneck = self.conv_block(f_down3)

        # print(f_down0.shape, f_jc0.shape)
        # print(f_down1.shape, f_jc1.shape)
        # print(f_down2.shape, f_jc2.shape)
        # print(f_down3.shape, f_jc3.shape)
        # print(f_bottleneck.shape)

        f_up0 = self.upsample_block0([f_jc3, f_bottleneck])
        f_up1 = self.upsample_block1([f_jc2, f_up0])
        f_up2 = self.upsample_block2([f_jc1, f_up1])
        f_up3 = self.upsample_block3([f_jc0, f_up2])

        # print('-'*20)
        # print(f_up0.shape)
        # print(f_up1.shape)
        # print(f_up2.shape)
        # print(f_up3.shape)

        ddf0 = self.ddf_fuse_layers[0](f_bottleneck)
        ddf1 = self.ddf_fuse_layers[1](f_up0)
        ddf2 = self.ddf_fuse_layers[2](f_up1)
        ddf3 = self.ddf_fuse_layers[3](f_up2)
        ddf4 = self.ddf_fuse_layers[4](f_up3)

        ddf = torch.sum(torch.stack([ddf0, ddf1, ddf2, ddf3, ddf4], axis=5), axis=5)
        return f_bottleneck, ddf
