import pdb

import numpy as np
import torch
from torch_scatter import scatter_min, segment_coo, scatter_mean, scatter_max


def obtain_voxel_size_based_grid_res(points_xyz, voxel_res=800):
    xyz_min, xyz_max = torch.min(points_xyz, dim=1)[0], torch.max(points_xyz, dim=1)[0]
    space_edge = torch.max(xyz_max - xyz_min) * 1.05
    voxel_size = space_edge*1.0/voxel_res
    # voxel_size_np = np.ones(3) * voxel_size
    return voxel_size, xyz_min


def construct_vox_points_closest(points_xyz, voxel_size=0.005, ranges=None):
    '''

    :param points_xyz: torch tensor, size (B X N x 3), where N = h*w
    :param voxel_size:
    :param ranges: torch tensor
    :return:
    '''
    batchsize = points_xyz.size(0)
    assert batchsize == 1
    # default sapce_min is None
    if ranges is None:
        xyz_min, xyz_max = torch.min(points_xyz, dim=1)[0], torch.max(points_xyz, dim=1)[0]
    else:
        xyz_min, xyz_max = ranges[:3], ranges[3:6]
    space_edge = torch.max(xyz_max - xyz_min) * 1.05
    xyz_shirt = points_xyz - xyz_min
    # voxel_size = torch.FloatTensor(voxel_size)
    sparse_id_list = []
    min_idx_list = []
    for bs in range(batchsize):
        xyz_shirt_item = xyz_shirt[bs]
        voxel_id_item = torch.floor(xyz_shirt_item/voxel_size).to(torch.int32)  # N x 3
        sparse_grid_idx, inv_idx = torch.unique(voxel_id_item, dim=0, return_inverse=True)

        # the xyz_centroid, centroid is "zhong xin"
        xyz_centroid = scatter_mean(points_xyz[bs], inv_idx, dim=0)
        xyz_centroid_prop = xyz_centroid[inv_idx, :]
        xyz_residual = torch.norm(points_xyz[bs] - xyz_centroid_prop, dim=-1)
        _, min_idx = scatter_min(xyz_residual, inv_idx, dim=0)
        #
        min_idx_list.append(min_idx.unsqueeze(0))
        sparse_id_list.append(sparse_grid_idx.unsqueeze(0))
    #
    min_idx_list = torch.cat(min_idx_list, dim=0)
    sparse_id_list = torch.cat(sparse_id_list, dim=0)
    # print("sparse_id", sparse_id_list.shape)
    # print("min_idx", min_idx_list.shape)
    # The min_idx is the point index of the grid voxel.

    # The xyz_centroid and sparse_grid_idx are useless !!!
    return sparse_id_list, min_idx_list


def incre_points(original_points_pos, incre_points_pos,  base_min, voxel_size, original_points_emb, incre_points_emb,):
    '''Combine the incremental points to original points'''
    if len(original_points_pos.size()) == 3:
        original_points_pos = original_points_pos[0]  # to N x 3
    if len(incre_points_pos.size()) == 3:
        original_points_pos = original_points_pos[0]  # to N x 3
    combined_points = torch.cat((original_points_pos, incre_points_pos), dim=0)
    # combined_emb = torch.cat((original_points_emb, incre_points_emb), dim=0)
    #
    c_points_shift = combined_points - base_min
    c_voxel_id_item = torch.floor(c_points_shift/voxel_size).to(torch.int32)
    c_sparse_grid_idx, c_inv_idx = torch.unique(c_voxel_id_item, dim=0, return_inverse=True)
    # select the points position
    xyz_centroid = scatter_mean(combined_points, c_inv_idx, dim=0)
    xyz_centroid_prop = xyz_centroid[c_inv_idx, :]
    xyz_residual = torch.norm(combined_points - xyz_centroid_prop, dim=-1)
    _, min_idx = scatter_min(xyz_residual, c_inv_idx, dim=0)
    selected_pos = torch.index_select(combined_points, dim=0, index=min_idx)
    # select the points embedding
    len_ori_points = original_points_pos.size(0)

    project_inv = c_inv_idx[:len_ori_points]
    project_inv_incre = c_inv_idx[len_ori_points:]
    try:
        selected_emb = _momentum_fusion(original_points_emb, project_inv, incre_points_emb, project_inv_incre)
    except:
        pdb.set_trace()
    return selected_pos, selected_emb, min_idx


def _momentum_fusion(points_emb, project_inv, incre_points_emb, incre_project_inv):
    ori_emb = scatter_mean(points_emb, project_inv, dim=0)
    incre_emb = scatter_mean(incre_points_emb, incre_project_inv, dim=0)
    ori_idx = (torch.sum(ori_emb, dim=1) > 0).to(torch.float32)
    incre_idx = (torch.sum(incre_emb, dim=1) > 0).to(torch.float32)
    len_ori = ori_idx.size(0)
    len_incre = incre_idx.size(0)
    if len_ori > len_incre:
        zeros_pad = torch.zeros((len_ori - len_incre)).to(points_emb.device)
        incre_idx = torch.cat((incre_idx, zeros_pad), dim=0)
        zeros_pad = torch.zeros((len_ori - len_incre), 32).to(points_emb.device)
        incre_emb = torch.cat([incre_emb, zeros_pad], dim=0)
    else:
        zeros_pad = torch.zeros((len_incre - len_ori)).to(points_emb.device)
        ori_idx = torch.cat((ori_idx, zeros_pad), dim=0)
        zeros_pad = torch.zeros((len_incre - len_ori), 32).to(points_emb.device)
        ori_emb = torch.cat([ori_emb, zeros_pad], dim=0)
    # pdb.set_trace()
    overlap_idx = incre_idx * ori_idx
    keep_same_idx_ori = ori_idx - overlap_idx
    keep_same_idx_incre = incre_idx - overlap_idx
    keep_same_idx_ori = keep_same_idx_ori.unsqueeze(1)
    keep_same_idx_incre = keep_same_idx_incre.unsqueeze(1)
    overlap_idx = overlap_idx.unsqueeze(1)
    final_emb = incre_emb * keep_same_idx_incre + ori_emb * keep_same_idx_ori + (ori_emb * overlap_idx * 0.9 + incre_emb * overlap_idx * 0.1)
    return final_emb
