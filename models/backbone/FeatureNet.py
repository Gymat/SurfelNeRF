import pdb

import torch
import torch.nn as nn
from inplace_abn import InPlaceABN
import torch.nn.functional as F
from models.backbone.mnasnet import mnasnet1_0_fixedBN
from .build import BACKBONE_REGISTRY


###################################  feature net  ######################################
@BACKBONE_REGISTRY.register()
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, cfg, intermediate=True, ):
        super(FeatureNet, self).__init__()
        norm_act = cfg.MODEL.IMG_ENC.NORM_ACT

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.intermediate = intermediate

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):

        B, V, _, H, W = x.shape
        x = x.reshape(B * V, 3, H, W)

        if self.intermediate:
            x1 = self.conv0(x)  # (B, 8, H, W)
            x2 = self.conv1(x1)  # (B, 16, H//2, W//2)
            x3 = self.conv2(x2)  # (B, 32, H//4, W//4)
            x3 = self.toplayer(x3)  # (B, 32, H//4, W//4)

            return [x, x1, x2, x3]
        else:
            # x: (B, 3, H, W)
            x = self.conv0(x) # (B, 8, H, W)
            x = self.conv1(x) # (B, 16, H//2, W//2)
            x = self.conv2(x) # (B, 32, H//4, W//4)
            x = self.toplayer(x) # (B, 32, H//4, W//4)

            return [x]

@BACKBONE_REGISTRY.register()
class MnasMulti(nn.Module):

    def __init__(self, cfg, alpha=1.0):
        super(MnasMulti, self).__init__()
        # depths = _get_depths(alpha)
        if alpha == 1.0:
            MNASNet = mnasnet1_0_fixedBN(pretrained=True, progress=True)
        else:
            raise NotImplementedError()
            # MNASNet = MNASNet(alpha=alpha)

        self.conv0 = nn.Sequential(
            MNASNet.layers._modules['0'],
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
            # MNASNet.layers._modules['8'],
        )
        # /2, channels = 16
        self.conv1 = MNASNet.layers._modules['8']
        # /4, channels = 24
        self.conv2 = MNASNet.layers._modules['9']
        # /8, channels = 40


    def forward(self, x):
        if len(x.size()) == 5:
            assert x.size(0) == 1
            x = x.squeeze(0)
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        return [x, conv0, conv1, conv2]  # 40 + 24 + 16 + 3 = 83


#############################################     MVS Net models        ################################################
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act='InPlaceABN', group_num=2):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        if norm_act == 'InPlaceABN':
            norm_act = InPlaceABN
            self.norm = norm_act(out_channels)
        elif norm_act == 'GN':
            norm_act = nn.GroupNorm
            self.norm = norm_act(group_num, out_channels)
        else:
            raise RuntimeError("norm_act is wrong!")
        # pdb.set_trace()
        # if issubclass(type(norm_act), InPlaceABN):
        #     self.norm = norm_act(out_channels)
        # elif issubclass(type(norm_act), nn.GroupNorm):
        #     self.norm = norm_act(out_channels, group_num)
        # else:
        #     raise RuntimeError("norm_act is wrong!")



    def forward(self, x):
        return self.norm(self.conv(x))

class ConvGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.GroupNorm, group_num=2):
        super(ConvGnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.gn = norm_act(out_channels, group_num)



    def forward(self, x):
        return self.gn(self.conv(x))


# =========================================  MNASNet modules  ==========================================================
def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]

if __name__ == '__main__':
    net = MnasMulti()
    a = torch.rand((1,3, 224, 224))
    o = net.forward(a)
    print(net)
    print(len(o))
    print(o[0].size())
    print(o[1].size())
    print(o[2].size())
