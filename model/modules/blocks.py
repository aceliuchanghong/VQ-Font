from functools import partial
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .modules import spectral_norm


# NOTE for nsml pytorch 1.1 docker
class Flatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


def dispatcher(dispatch_fn):
    def decorated(key, *args):
        if callable(key):
            return key

        if key is None:
            key = 'none'

        return dispatch_fn(key, *args)

    return decorated


@dispatcher
def norm_dispatch(norm):
    return {
        # 不改变输入数据，直接输出 常用于保持模型结构的一致性，在需要用到占位符或者测试时比较有用
        'none': nn.Identity,
        # nn.InstanceNorm2d 是一种归一化层，它对每个输入样本的每个通道独立地进行归一化处理
        # affine：如果设置为 True，该层将具有可学习的缩放和平移参数。如果设置为 False，则直接进行归一化而不进行缩放和平移
        # InstanceNorm2d 在图像生成任务中尤其常见，如风格迁移（Style Transfer）和生成对抗网络（GANs）。与批归一化不同，它在处理单张图片时表现更好，因为它不会受到批量大小的影响，也不依赖其他样本的分布。
        'in': partial(nn.InstanceNorm2d, affine=False),
        'bn': nn.BatchNorm2d,  # nn.BatchNorm2d 是PyTorch中的批归一化层，广泛用于加速神经网络训练并稳定其训练过程
    }[norm.lower()]


@dispatcher
def w_norm_dispatch(w_norm):
    # NOTE Unlike other dispatcher, w_norm is function, not class.
    return {
        'spectral': spectral_norm,
        'none': lambda x: x
    }[w_norm.lower()]


@dispatcher
def activ_dispatch(activ, norm=None):
    return {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
    }[activ.lower()]


@dispatcher
def pad_dispatch(pad_type):
    return {
        "zero": nn.ZeroPad2d,
        "replicate": nn.ReplicationPad2d,
        "reflect": nn.ReflectionPad2d
    }[pad_type.lower()]


class ConvBlock(nn.Module):
    """ pre-active conv block """

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, norm='none',
                 activ='relu', bias=True, upsample=False, downsample=False, w_norm='none',
                 pad_type='zero', dropout=0.0, size=None):
        # 1x1 conv assertion
        if kernel_size == 1:
            assert padding == 0
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out

        activ = activ_dispatch(activ, norm)
        norm = norm_dispatch(norm)
        w_norm = w_norm_dispatch(w_norm)
        pad = pad_dispatch(pad_type)
        self.upsample = upsample
        self.downsample = downsample

        self.norm = norm(C_in)
        self.activ = activ()

        if dropout > 0.:
            self.dropout = nn.Dropout2d(p=dropout)
        self.pad = pad(padding)
        self.conv = w_norm(nn.Conv2d(C_in, C_out, kernel_size, stride, bias=bias))

    def forward(self, x):
        x = self.norm(x)
        x = self.activ(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.conv(self.pad(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x


class ResBlock(nn.Module):
    """ Pre-activate ResBlock with spectral normalization """

    def __init__(self, C_in, C_out, kernel_size=3, padding=1, upsample=False, downsample=False,
                 norm='none', w_norm='none', activ='relu', pad_type='zero', dropout=0.,
                 scale_var=False):
        assert not (upsample and downsample)
        super().__init__()
        w_norm = w_norm_dispatch(w_norm)
        self.C_in = C_in
        self.C_out = C_out
        self.upsample = upsample
        self.downsample = downsample
        self.scale_var = scale_var

        self.conv1 = ConvBlock(C_in, C_out, kernel_size, 1, padding, norm, activ,
                               upsample=upsample, w_norm=w_norm, pad_type=pad_type,
                               dropout=dropout)
        self.conv2 = ConvBlock(C_out, C_out, kernel_size, 1, padding, norm, activ,
                               w_norm=w_norm, pad_type=pad_type, dropout=dropout)

        # XXX upsample / downsample needs skip conv?
        if C_in != C_out or upsample or downsample:
            self.skip = w_norm(nn.Conv2d(C_in, C_out, 1))

    def forward(self, x):
        """
        normal: pre-activ + convs + skip-con
        upsample: pre-activ + upsample + convs + skip-con
        downsample: pre-activ + convs + downsample + skip-con
        => pre-activ + (upsample) + convs + (downsample) + skip-con
        """
        out = x

        out = self.conv1(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        # skip-con
        if hasattr(self, 'skip'):
            if self.upsample:
                x = F.interpolate(x, scale_factor=2)
            x = self.skip(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)

        out = out + x
        if self.scale_var:
            out = out / np.sqrt(2)
        return out
