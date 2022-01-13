import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def init_weights(modules):
    for module in modules():
        if isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight.data)


# ========================= Basic component ==============================
class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class ResidualBlock(nn.Module):
    '''
    Residual block for DRLN+
    Composed by two Conv. and two ReLU.
    '''
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        init_weights(self.modules)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out + x)
        return out


class BasicBlockReLU(nn.Module):
    '''
    Basic block with relu activation for DRLN+
    Composed by Conv2d + ReLU
    '''
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1,
                 dilation=1):
        super(BasicBlockReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad,
                              dilation)
        self.relu = nn.ReLU(inplace=True)

        init_weights(self.modules)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out


class BasicBlockSigmoid(nn.Module):
    '''
    Basic block with sigmoid activation
    Composed by Conv2d + sigmoid
    '''
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlockSigmoid, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad)
        self.sigmoid = nn.Sigmoid()

        init_weights(self.modules)

    def forward(self, x):
        out = self.conv(x)
        out = self.sigmoid(out)
        return out


# ================== Dense Residual Laplacian Module =====================
class LPA(nn.Module):
    '''
    Laplacian Pyramid Attention (LPA)
    Composed by:
        Avgerage pooling
        Feature reduction operator D3, D5, D7 (Pyramid here)
        Upsampling operator + sigmoid
    '''
    def __init__(self, channel, reduction=16):
        super(LPA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.D3 = BasicBlockReLU(channel, channel // reduction, 3, 1, 3, 3)
        self.D5 = BasicBlockReLU(channel, channel // reduction, 3, 1, 5, 5)
        self.D7 = BasicBlockReLU(channel, channel // reduction, 3, 1, 7, 7)

        self.up = BasicBlockSigmoid((channel // reduction) * 3, channel,
                                    3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        d3 = self.D3(y)
        d5 = self.D5(y)
        d7 = self.D7(y)
        c_out = torch.cat([d3, d5, d7], dim=1)
        y = self.up(c_out)
        return x * y


class DRLM(nn.Module):
    '''
    Dense Residual Laplacian Module (DRLM)
    Composed by:
        Three residual block with dense connection
        Compression unit
        Laplacian pyramid attention
    '''
    def __init__(self, in_channels, out_channels):
        super(DRLM, self).__init__()
        self.residual1 = ResidualBlock(in_channels, out_channels)
        self.residual2 = ResidualBlock(in_channels*2, out_channels*2)
        self.residual3 = ResidualBlock(in_channels*4, out_channels*4)
        self.compression = BasicBlockReLU(in_channels*8, out_channels, 1, 1, 0)
        self.attention = LPA(in_channels)

    def forward(self, x):
        c0 = x

        r1 = self.residual1(c0)
        c1 = torch.cat([c0, r1], dim=1)

        r2 = self.residual2(c1)
        c2 = torch.cat([c1, r2], dim=1)

        r3 = self.residual3(c2)
        c3 = torch.cat([c2, r3], dim=1)
        g = self.compression(c3)
        out = self.attention(g)
        return out


# ================== Upsampling =============================
class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2)
            self.up3 = _UpsampleBlock(n_channels, scale=3)
            self.up4 = _UpsampleBlock(n_channels, scale=4)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self,  n_channels, scale):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1),
                            nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1),
                        nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


# ======================== Reconstruction block ==========================
class ReconstructBlock(nn.Module):
    def __init__(self, channel, multi_scale):
        super(ReconstructBlock, self).__init__()

        if multi_scale:
            self.rec2 = nn.Conv2d(channel, 3, 3, 1, 1)
            self.rec3 = nn.Conv2d(channel, 3, 3, 1, 1)
            self.rec4 = nn.Conv2d(channel, 3, 3, 1, 1)
        else:
            self.rec = nn.Conv2d(channel, 3, 3, 1, 1)
        self.multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.rec2(x)
            if scale == 3:
                return self.rec3(x)
            if scale == 4:
                return self.rec4(x)

        return self.rec(x)
