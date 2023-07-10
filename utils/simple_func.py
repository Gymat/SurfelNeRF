import os
import torch
from skimage.metrics import structural_similarity
import numpy as np


'''
Utility func
'''

img2mse = lambda x, y : np.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * np.log(x) / np.log(np.full((1,), 10.,))


def strlist_to_floatlist(s):
    num_list = s.strip().split(" ")
    # print(num_list)
    float_list = [float(x) for x in num_list]
    return float_list

def str_to_strlist(s):
    num_list = s.strip().split(" ")
    # print(num_list)
    # float_list = [x for x in num_list]
    return num_list


'''
Metrics
'''
def pad_edge(img, pd):
    # Only for scannet
    img[:pd, :, :] = 0
    img[-pd:, :, :] = 0
    img[:, :pd, :] = 0
    img[:, -pd:, :] = 0
    return img


def compute_metrics(rgb, target, lpips_alex=None, lpips_vgg=None, pad_edge_value=8):
    '''

    :param rgb: np.array H x W x 3
    :param target:
    :return:
    '''
    rgb = pad_edge(rgb, pad_edge_value)
    target = pad_edge(target, pad_edge_value)
    img_loss = img2mse(rgb, target)
    psnr = mse2psnr(img_loss)
    ssim_1 = structural_similarity(rgb, target, channel_axis=-1, data_range=1.0,
                                 multichannel=True)
    ssim = structural_similarity(rgb, target, channel_axis=-1,
                                   multichannel=True)
    rgb = torch.from_numpy(rgb)
    target = torch.from_numpy(target)
    # rgb = rgb * 2 - 1  # make the range in [-1,1]
    # target = target * 2 - 1
    lpips_a = lpips_alex(rgb.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]
    lpips_v = lpips_vgg(rgb.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]
    metrics = {"img_loss": img_loss.item(), "psnr": psnr.item(), "ssim_1": ssim_1, "ssim": ssim, "lpips_alex": lpips_a[0, 0, 0],
               "lpips_vgg": lpips_v[0, 0, 0], }
    return metrics

