import pdb

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

"""
This class provide differentable depth to point position
"""

class Depth2Point(nn.Module):
    def __init__(self, cfg):
        super(Depth2Point, self).__init__()
        self.height, self.width = cfg.img_size_h, cfg.img_size_w
        self.depth_trunc = 15.0
        #
        self.cam_pos_cam_coord = np.asarray([0, 0, 0])
        # init useful params
        # We first build the coordinate matrix, which size is (H, W, 2), the value is the (x,y) coordinate of corresponding index
        i_coords, j_coords = np.meshgrid(range(self.height), range(self.width), indexing='ij')
        self.coord_mat = torch.FloatTensor(np.concatenate((j_coords[..., None], i_coords[..., None]), axis=-1)).to(cfg.DEVICE)
        self.one_mat = torch.FloatTensor(np.ones((self.height, self.width, 1))).to(cfg.DEVICE)
        #
        self.coord_mat.to('cuda')
        self.one_mat.to('cuda')

    def project(self, intrinsic_depth, depth_map, extrinsic_c2w_matrix):
        '''

        :param intrinsic_depth: (B, 3, 3)
        :param depth_map: (B, H, W)
        :param extrinsic_c2w_matrix: (B, 4, 4)
        :return:
        '''
        B, H, W = depth_map.size()
        assert H == self.height
        assert W == self.width
        assert B == 1
        # Setup about intrinsic matrix
        #
        with torch.no_grad():
            focal_length = (intrinsic_depth[0, 0, 0], intrinsic_depth[0, 1, 1],)
            principal_point = (intrinsic_depth[0, 0, 2], intrinsic_depth[0, 1, 2])
            principal_point_mat = torch.Tensor([principal_point[0], principal_point[1]])
            principal_point_mat = principal_point_mat.reshape(1, 1, 2)
            principal_point_mat = principal_point_mat.to('cuda')
            focal_length_mat = torch.Tensor([focal_length[0], focal_length[1]])
            focal_length_mat = focal_length_mat.reshape(1, 1, 2)
            focal_length_mat = focal_length_mat.to('cuda')
            #

            depth_map = depth_map.permute((1, 2, 0))
            #
            masked_depth_map = torch.zeros_like(depth_map)
            masked_depth_map[(depth_map < self.depth_trunc) & (depth_map > 0)] = 1  # valid is 1!
            #
            point_mask = masked_depth_map  # h, w, 1
            point_mask[:6, ...] = 0
            point_mask[-6:, ...] = 0
            point_mask[:, :6,] = 0
            point_mask[:, -6:] = 0
            # pdb.set_trace()
            cam_coord_part = (self.coord_mat - principal_point_mat) / focal_length_mat
        # do the computation
        cam_coord_xy = cam_coord_part * depth_map
        cam_coord_xyz = torch.cat([cam_coord_xy, depth_map], dim=-1)  # (h, w, 3)
        cam_coord_xyz1 = torch.cat([cam_coord_xyz, self.one_mat], dim=-1)  # (h, w, 4)
        #
        xyz1_cam_coord = cam_coord_xyz1.permute((2, 0, 1))  # (4, h, w)
        xyz1_cam_coord = xyz1_cam_coord.reshape(4, self.height * self.width)  # (4, h*w)
        # print(xyz1_cam_coord.shape, c2w_mat.shape)
        world_coord_mat = torch.matmul(extrinsic_c2w_matrix, xyz1_cam_coord)  # (4, h*w)
        world_coord_mat = world_coord_mat[0]
        # pdb.set_trace()
        # print(world_coord_mat.shape)
        world_coord_mat = world_coord_mat[:3, :]  # (3, h*w)
        world_coord_mat = world_coord_mat.T  # (h*w, 3)
        # points_dir_world_coord = self.obtain_point_dir(cam_coord_xyz, extrinsic_c2w_matrix)  # (h*w, 3)
        # reshape the size of point_mask from (h, w, 1) to (h*w, 1)
        point_mask = point_mask.reshape(self.height * self.width, 1)
        # print(world_coord_mat.shape)
        return world_coord_mat, point_mask

    def obtain_point_dir(self, cam_coord_xyz, extrinsic_c2w_m):
        h, w, _ = cam_coord_xyz.shape
        points_dir = cam_coord_xyz  # (h, w, 3)
        points_dir = points_dir.reshape(h*w, 3)
        points_dir_normed = F.normalize(points_dir, dim=-1, p=2)
        points_dir_normed = points_dir_normed.T  # (3, h*w)
        # change the dir under world coordinates
        cam2world_mat3x3 = extrinsic_c2w_m[:3, :3]
        points_dir_world_coor = torch.matmul(cam2world_mat3x3, points_dir_normed)  # (3, h*w)
        # points_dir_world_coor = points_dir_world_coor.reshape((3, h, w))
        # points_dir_world_coor = np.transpose(points_dir_world_coor, (1, 2, 0))  # ()
        return points_dir_world_coor.T  # (h*w, 3)


