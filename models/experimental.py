# YOLOv5 experimental modules
import math

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv, DWConv
from utils.google_utils import attempt_download

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = False  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

#------------------------------------ SR -------------------------------------------------

class default_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True): # ch_in, ch_out, kernel, stride, padding, groups
        super(default_conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        return self.conv(x)

class MeanShift_conv(nn.Conv2d):
    def __init__(self, c1, c2, sign=-1, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0)):
        super(MeanShift_conv, self).__init__(c1, c2, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class MeanShift(nn.Module):
    def __init__(self, c1, c2, sign=-1):
        super(MeanShift, self).__init__()
        self.rgb_mean = (0.4488, 0.4371, 0.4040)
        self.rgb_std = (1.0, 1.0, 1.0)
        self.rgb_range = 255
        self.shift_conv = MeanShift_conv(c1, c2, sign, rgb_range=self.rgb_range, rgb_mean=self.rgb_mean, rgb_std=self.rgb_std)

    def forward(self, x):
        return self.shift_conv(x)

class ResBlock(nn.Module):
    def __init__(self, c1, c2, res_scale=1):
        super(ResBlock, self).__init__()
        self.cv1 = default_conv(c1, c2, 3, 1)
        self.act = nn.ReLU(True)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.cv1(self.act(self.cv1(x))).mul(self.res_scale)
        res = res + x
        return res

class Upsampler(nn.Module):
    def __init__(self, c1, c2, scale):
        super(Upsampler, self).__init__()
        self.scale = scale
        if (scale & (scale - 1)) == 0:
            self.cv1 = default_conv(c1, c2 * 4, 3, 1)
            self.pixel = nn.PixelShuffle(2)
        elif scale == 3:
            self.cv1 = default_conv(c1, c2 * 9, 3) # 9 * c2
            self.pixel = nn.PixelShuffle(3)
        else:
            raise NotImplementedError
    def forward(self, x):
        if (self.scale & (self.scale - 1)) == 0:
            for _ in range(int(math.log(self.scale, 2))):
                x = self.cv1(x)
                x = self.pixel(x)
        elif self.scale == 3:
            x = self.cv1(x)
            x = self.pixel(x)
        else:
            raise NotImplementedError
        return x

class Add(nn.Module):
    def __init__(self, n):
        super(Add, self).__init__()
        self.iter = range(n - 1)

    def forward(self, x):
        y = x[0]
        for i in self.iter:
            y = y + x[i + 1]
        return y

# def default_conv1(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size//2), bias=bias)

# class ResBlock1(nn.Module):
#     def __init__(self, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
#         super(ResBlock1, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(default_conv1(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: m.append(nn.BatchNorm2d(n_feat))
#             if i == 0: m.append(act)
#
#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x
#
#         return res

# class Upsampler1(nn.Sequential):
#     def __init__(self, n_feat, scale, bn=False, act=False, bias=True):
#
#         m = []
#         if (scale & (scale - 1)) == 0:    # Is scale = 2^n? 2^n일 경우 bit 가 10000 식인데 여기서 1 빼면 01111 이라 & 하면 겹치는게 없어 0이 나옴
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(default_conv1(n_feat, 4 * n_feat, 3, bias))
#                 m.append(nn.PixelShuffle(2))
#                 if bn: m.append(nn.BatchNorm2d(n_feat))
#                 if act: m.append(act())
#         elif scale == 3:
#             m.append(default_conv1(n_feat, 9 * n_feat, 3, bias))
#             m.append(nn.PixelShuffle(3))
#             if bn: m.append(nn.BatchNorm2d(n_feat))
#             if act: m.append(act())
#         else:
#             raise NotImplementedError
#
#         super(Upsampler1, self).__init__(*m)

