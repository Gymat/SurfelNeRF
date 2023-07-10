import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def volumetric_rendering(rgb, density, dists,):
    '''

    :param rgb:  [B, R, SR, K, 3], where R is ray, SR is the shading points
    :param density: [B, R, SR, K]
    :param depth_dist: torch.Size([B, R, K])
    :return:
    '''
    eps = 1e-10
    # t_vals = depth_dist
    # dists = t_vals[..., 1:] - t_vals[..., :-1]
    # dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    #
    alpha = 1.0 - torch.exp(-density * dists)
    # cumprod exclusive
    T = torch.cumprod(1.0 - alpha + eps, dim=-1)
    temp = torch.ones_like(alpha[..., :1])
    accum_prod = torch.cat([
        temp, T[..., :-1]
    ], dim=-1)
    # blend weights
    weights = alpha * accum_prod

    # output
    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    # depth = (weights * t_vals)
    acc = weights.sum(dim=-1)

    return comp_rgb, acc, weights


def get_t_vals(depth_dist, valid_pnt_mask, filter_mask=True, min_dis=0.003, far=400):
    t_vals = depth_dist
    # pdb.set_trace()
    t_vals = torch.masked_fill(t_vals, ~valid_pnt_mask, far)
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    if filter_mask:
        mask = dists < 1e-8
        mask = mask.to(torch.float32)
        dists = dists * (1 - mask) + mask * min_dis
    valid_dists = dists * valid_pnt_mask
    return valid_dists


def fill_bg(ray_mask, ray_color, bg_color, **kwargs):
    # check bg_color
    if len(bg_color.size()) == 1:
        bg_color = bg_color.unsqueeze(0)
    # assert len(bg_color.size()) == 2  # 1 x 3
    B, R = ray_mask.shape
    ray_inds = torch.nonzero(ray_mask)
    #
    # pdb.set_trace()
    final_color = torch.ones_like(ray_color) * bg_color[None, ...]
    final_color[ray_inds[..., 0], ray_inds[..., 1], :] = ray_color[ray_inds[..., 0], ray_inds[..., 1], :]

    return final_color

def fill_bg_RGBGT(ray_mask, ray_color, bg_color, rgb_GT, replace_GT_with_bg=False):
    # check bg_color
    if len(bg_color.size()) == 1:
        bg_color = bg_color.unsqueeze(0)
    # assert len(bg_color.size()) == 2  # 1 x 3
    B, R = ray_mask.shape
    ray_inds = torch.nonzero(ray_mask)
    #
    # pdb.set_trace()
    final_color = rgb_GT
    if replace_GT_with_bg:
        final_color1 = torch.ones_like(rgb_GT) * bg_color[None, ]
    else:
        final_color1 = final_color
    final_color1[ray_inds[..., 0], ray_inds[..., 1], :] = ray_color[ray_inds[..., 0], ray_inds[..., 1], :]

    return final_color1


def split_depth_value(depth):
    pass
