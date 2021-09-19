import torch.nn as nn
import torch
import src.model.layers as layers


class LocalModel(nn.Module):
    def __init__(self, input_shape, nc_initial=16):
        super(LocalModel, self).__init__()
        self.input_shape = input_shape
        self.ddf_levels = [0, 1, 2, 3, 4]

        nc = [nc_initial*(2**i) for i in range(5)]

        self.downsample_block0 = layers.DownsampleBlock(inc=2, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.conv_block = layers.Conv3dBlock(inc=nc[3], outc=nc[4])

        self.upsample_block0 = layers.UpSampleBlock(inc=nc[4], outc=nc[3], output_size=(13, 13, 11))
        self.upsample_block1 = layers.UpSampleBlock(inc=nc[3], outc=nc[2], output_size=(26, 26, 23))
        self.upsample_block2 = layers.UpSampleBlock(inc=nc[2], outc=nc[1], output_size=(52, 52, 46))
        self.upsample_block3 = layers.UpSampleBlock(inc=nc[1], outc=nc[0], output_size=(104, 104, 92))

        self.ddf_fuse_layers = [layers.DDFFusion(inc=nc[4 - i], out_shape=self.input_shape).cuda() for i in self.ddf_levels]

    def forward(self, x):
        f_down0, f_jc0 = self.downsample_block0(x)
        f_down1, f_jc1 = self.downsample_block1(f_down0)
        f_down2, f_jc2 = self.downsample_block2(f_down1)
        f_down3, f_jc3 = self.downsample_block3(f_down2)
        f_bottleneck = self.conv_block(f_down3)

        f_up0 = self.upsample_block0([f_jc3, f_bottleneck])
        f_up1 = self.upsample_block1([f_jc2, f_up0])
        f_up2 = self.upsample_block2([f_jc1, f_up1])
        f_up3 = self.upsample_block3([f_jc0, f_up2])

        ddf0 = self.ddf_fuse_layers[0](f_bottleneck)
        ddf1 = self.ddf_fuse_layers[1](f_up0)
        ddf2 = self.ddf_fuse_layers[2](f_up1)
        ddf3 = self.ddf_fuse_layers[3](f_up2)
        ddf4 = self.ddf_fuse_layers[4](f_up3)

        ddf = torch.mean(torch.stack([ddf0, ddf1, ddf2, ddf3, ddf4], axis=5), axis=5)
        return f_bottleneck, ddf


class MultiTaskModel(nn.Module):
    def __init__(self, input_shape, nc_initial=16):
        super(MultiTaskModel, self).__init__()
        self.input_shape = input_shape
        self.ddf_levels = [0, 1, 2, 3, 4]

        nc = [nc_initial*(2**i) for i in range(5)]

        self.downsample_block0 = layers.DownsampleBlock(inc=2, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.conv_block = layers.Conv3dBlock(inc=nc[3], outc=nc[4])

        # ddf branch (b0)
        self.upsample_block0_b0 = layers.UpSampleBlock(inc=nc[4], outc=nc[3], output_size=(13, 13, 11))
        self.upsample_block1_b0 = layers.UpSampleBlock(inc=nc[3], outc=nc[2], output_size=(26, 26, 23))
        self.upsample_block2_b0 = layers.UpSampleBlock(inc=nc[2], outc=nc[1], output_size=(52, 52, 46))
        self.upsample_block3_b0 = layers.UpSampleBlock(inc=nc[1], outc=nc[0], output_size=(104, 104, 92))

        # segmentation branch (b1)
        self.upsample_block0_b1 = layers.UpSampleBlock(inc=nc[4], outc=nc[3], output_size=(13, 13, 11))
        self.upsample_block1_b1 = layers.UpSampleBlock(inc=nc[3], outc=nc[2], output_size=(26, 26, 23))
        self.upsample_block2_b1 = layers.UpSampleBlock(inc=nc[2], outc=nc[1], output_size=(52, 52, 46))
        self.upsample_block3_b1 = layers.UpSampleBlock(inc=nc[1], outc=nc[0], output_size=(104, 104, 92))

        self.ddf_fuse_layers = [layers.DDFFusion(inc=nc[4 - i], out_shape=self.input_shape).cuda() for i in self.ddf_levels]

    def forward(self, x):
        f_down0, f_jc0 = self.downsample_block0(x)
        f_down1, f_jc1 = self.downsample_block1(f_down0)
        f_down2, f_jc2 = self.downsample_block2(f_down1)
        f_down3, f_jc3 = self.downsample_block3(f_down2)
        f_bottleneck = self.conv_block(f_down3)

        # ddf branch (b0)
        f_up0_b0 = self.upsample_block0_b0([f_jc3, f_bottleneck])
        f_up1_b0 = self.upsample_block1_b0([f_jc2, f_up0_b0])
        f_up2_b0 = self.upsample_block2_b0([f_jc1, f_up1_b0])
        f_up3_b0 = self.upsample_block3_b0([f_jc0, f_up2_b0])

        ddf0 = self.ddf_fuse_layers[0](f_bottleneck)
        ddf1 = self.ddf_fuse_layers[1](f_up0_b0)
        ddf2 = self.ddf_fuse_layers[2](f_up1_b0)
        ddf3 = self.ddf_fuse_layers[3](f_up2_b0)
        ddf4 = self.ddf_fuse_layers[4](f_up3_b0)
        ddf = torch.mean(torch.stack([ddf0, ddf1, ddf2, ddf3, ddf4], axis=5), axis=5)

        # segmentation branch (b1)s
        f_up0_b1 = self.upsample_block0_b1([f_jc3, f_bottleneck])
        f_up1_b1 = self.upsample_block1_b1([f_jc2, f_up0_b1])
        f_up2_b1 = self.upsample_block2_b1([f_jc1, f_up1_b1])
        f_up3_b1 = self.upsample_block3_b1([f_jc0, f_up2_b1])
        seg = torch.sigmoid(f_up3_b1)

        return f_bottleneck, ddf, seg