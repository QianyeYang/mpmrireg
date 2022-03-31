import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class Conv3dBlock(nn.Module):
    def __init__(self, inc, outc, ksize=3, stride=1, pad=1):
        super(Conv3dBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=inc, out_channels=outc,
                              kernel_size=ksize, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(outc)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Deconv3d(nn.Module):
    def __init__(self, inc, outc, output_size, ksize=2, stride=2, pad=0, dilation=1):
        super(Deconv3d, self).__init__()
        self.output_size = output_size
        self.pad = pad
        self.deconv = nn.ConvTranspose3d(
            in_channels=inc, out_channels=outc, kernel_size=ksize, stride=stride,
            padding=pad, dilation=dilation, output_padding=0)
        self.bn = nn.BatchNorm3d(outc)
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.upsample_layer = nn.Upsample(size=output_size, mode='trilinear', align_corners=True)

        nn.init.xavier_uniform_(self.deconv.weight)

    def forward(self, x):
        out = self.act(self.bn(self.deconv(x)))
        if out.shape[-3:] != self.output_size:
            out = self.upsample_layer(out)
        return out


class AdaptiveUpsample3d(nn.Module):
    def __init__(self, inc, outc, output_size):
        super(AdaptiveUpsample3d, self).__init__()
        self.upsample_layer = nn.Upsample(size=output_size, mode='trilinear', align_corners=True)
        self.conv = nn.Conv3d(inc, outc, kernel_size=1, stride=1, padding=0)

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        output = self.upsample_layer(x)
        output = self.conv(output)
        return output


class DDFFusion(nn.Module):
    def __init__(self, inc, out_shape):
        super(DDFFusion, self).__init__()
        self.conv = nn.Conv3d(in_channels=inc, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.out_shape = out_shape

        self.conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv.weight.shape))
        # nn.init.xavier_uniform(self.conv.weight)
        self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))

    def forward(self, x):
        output = self.conv(x)
        if output.shape[-3:] != self.out_shape:
            output = F.interpolate(output, size=self.out_shape, mode='trilinear', align_corners=True)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, channels, ksize=3, stride=1, pad=1):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = Conv3dBlock(inc=channels, outc=channels, ksize=ksize, stride=stride, pad=pad)
        self.conv_block2 = Conv3dBlock(inc=channels, outc=channels, ksize=ksize, stride=stride, pad=pad)

    def forward(self, x):
        output = self.conv_block1(x)
        output = self.conv_block2(output)
        return output + x


class DownsampleBlock(nn.Module):
    def __init__(self, inc, outc, ksize=3, stride=1, pad=1):
        super(DownsampleBlock, self).__init__()
        self.conv = Conv3dBlock(inc=inc, outc=outc, ksize=ksize, stride=stride, pad=pad)
        self.resblock = ResidualBlock(channels=outc, ksize=ksize, stride=stride, pad=pad)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        f_jc = self.conv(x)
        f_down = self.resblock(f_jc)
        f_down = self.max_pool(f_down)
        return f_down, f_jc


class UpSampleBlock(nn.Module):
    def __init__(self, inc, outc, output_size):
        super(UpSampleBlock, self).__init__()
        self.deconv = Deconv3d(inc=inc, outc=outc, output_size=output_size)
        self.adpt_up = AdaptiveUpsample3d(inc=inc, outc=outc, output_size=output_size)
        self.conv1 = Conv3dBlock(inc=outc, outc=outc)
        self.conv2 = Conv3dBlock(inc=outc, outc=outc)

    def forward(self, x):
        jc_feature, ori_feature = x[0], x[1]
        tmp = self.deconv(ori_feature) + self.adpt_up(ori_feature)
        res_feature = tmp + jc_feature
        tmp = self.conv1(tmp)
        tmp = self.conv2(tmp)
        return res_feature + tmp