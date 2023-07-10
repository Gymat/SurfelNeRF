import select

import torch
import torch.nn as nn
import os
import numpy as np
from ..helpers.networks import init_seq, positional_encoding
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune_param
from .query_point_indices_worldcoords_ext import lighting_fast_querier as lighting_fast_querier_w
from utils.simple_func import strlist_to_floatlist
from models.neural_points.build import POINTS_REGISTRY
from models.point_con.voxelize import incre_points
import pdb



@POINTS_REGISTRY.register()
class NeuralPoints_Simple(nn.Module):
    def __init__(self, cfg, **kwargs, ):
        super().__init__()
        self.opt = cfg
        self.device = torch.device("cuda:0")
        device = torch.device("cuda:0")
        print("Device: ", self.device, " ", device)
        # set parameters
        self.test_ = torch.rand((1,1,1), device=device, dtype=torch.float32)

        kernel_size = strlist_to_floatlist(cfg.MODEL.NEURAL_POINTS.KERNEL_SIZE)
        self.query_size = kernel_size
        if cfg.MODEL.NEURAL_POINTS.WCOOR_QUERY:
            self.lighting_fast_querier = lighting_fast_querier_w
        else:
            raise NotImplemented()
        # self.lighting_fast_querier = lighting_fast_querier_w if self.opt.wcoord_query > 0 else lighting_fast_querier_p
        self.querier = self.lighting_fast_querier(device, self.opt)

    def forward(self, inputs, **kwargs):
        '''

        :param inputs:
        :return:
        '''
        """ 
        First prepare the input for this func!
        """
        # setup for points input
        points_info = inputs['points_info']
        point_cloud_pos = points_info['points_position']
        points_embeddings = points_info['points_embeddings']
        points_color = points_info['points_color']
        points_dir = points_info['points_dir']
        # setup for raydir
        ray_dir_tensor = inputs["raydir"]
        # Setup for other params
        camrotc2w, campos, near_plane, far_plane, intrinsic = inputs["camrotc2w"], inputs["campos"], \
                                                                    inputs["near"], inputs["far"], \
                                                                    inputs["intrinsic"]
        # pixel_idx = pixel_idx.to(self.device)

        # 1, 294, 24, 32;   1, 294, 24;     1, 291, 2

        """
        Do the computation!
        """

        # The bug: sample_pidx is size [:, 0, ...]
        if len(point_cloud_pos.size()) == 2:
            point_cloud_pos = point_cloud_pos.unsqueeze(0)
        output_tensor = self.get_point_indices(
            ray_dir_tensor, camrotc2w, campos, torch.min(near_plane).cpu().numpy(),
            torch.max(far_plane).cpu().numpy(),
            point_cloud_pos)  # default opt.NN = 2, so vox_query is false

        # output_tensor['sample_pidx'] = sample_pidx_tensor
        #         output_tensor['sample_loc_cam_coor'] = sample_loc_cam_coor_tensor
        #         output_tensor['ray_mask'] = ray_mask_tensor
        #         output_tensor['point_xyz_pers'] = point_xyz_pers_tensor
        #         output_tensor['sample_loc_w_coor'] = sample_loc_world_coor_tensor
        #         output_tensor['sample_ray_dirs'] = sample_ray_dirs_tensor
        #         output_tensor['visze'] = vsize
        # obtain target params from output_tensor
        sample_pidx = output_tensor['sample_pidx']
        # point_xyz_pers_tensor = output_tensor['point_xyz_pers']



        sample_pnt_mask = sample_pidx >= 0
        B, R, SR, K = sample_pidx.shape
        # B: batch, R: ray number, K: max neural points each group,
        # SR: max shading points number each ray
        # if R == 0:
        #     pdb.set_trace()
        sample_pidx = torch.clamp(sample_pidx, min=0).view(-1).long()

        # For sampled embedding
        # TODO: NOTE!!! I emit the position in the camera coordinate!
        sampled_embedding_o = torch.index_select(
            torch.cat([point_cloud_pos, points_embeddings[None, ...]], dim=-1), 1, sample_pidx)
        sampled_embedding = sampled_embedding_o.view(B, R, SR, K, points_embeddings.shape[-1] + point_cloud_pos.shape[-1])
        # For colors
        sampled_color = torch.index_select(points_color, 0, sample_pidx).view(B, R, SR, K, points_color.shape[-1])
        # For dir
        sampled_dir = torch.index_select(points_dir, 0, sample_pidx).view(B, R, SR, K, points_dir.shape[-1])
        # For conf
        sampled_conf = None

        # filepath = "./sampled_xyz_full.txt"
        # np.savetxt(filepath, self.xyz.reshape(-1, 3).detach().cpu().numpy(), delimiter=";")
        #
        # filepath = "./sampled_xyz_pers_full.txt"
        # np.savetxt(filepath, point_xyz_pers_tensor.reshape(-1, 3).detach().cpu().numpy(), delimiter=";")

        # if self.xyz.grad is not None:
        #     print("xyz grad:", self.xyz.requires_grad, torch.max(self.xyz.grad), torch.min(self.xyz.grad))
        # if self.points_embeding.grad is not None:
        #     print("points_embeding grad:", self.points_embeding.requires_grad, torch.max(self.points_embeding.grad))
        # print("points_embeding 3", torch.max(self.points_embeding), torch.min(self.points_embeding))
        """ Put the value in the output dict"""
        output_sampled_dict = dict()
        output_sampled_dict['sampled_color'] = sampled_color
        # output_sampled_dict['sampled_Rw2c'] = sampled_Rw2c
        output_sampled_dict['sampled_dir'] = sampled_dir
        output_sampled_dict['sampled_conf'] = sampled_conf
        output_sampled_dict['sampled_embedding'] = sampled_embedding[..., 3:]
        # output_sampled_dict['sampled_xyz_pers'] = sampled_embedding[..., 3:6]
        output_sampled_dict['sampled_xyz'] = sampled_embedding[..., :3]
        output_sampled_dict['sampled_pnt_mask'] = sample_pnt_mask
        output_sampled_dict['sample_loc_cam_coor'] = output_tensor['sample_loc_cam_coor']
        output_sampled_dict['sample_loc_w_coor'] = output_tensor['sample_loc_w_coor']
        output_sampled_dict['sample_ray_dirs'] = output_tensor['sample_ray_dirs']
        output_sampled_dict['ray_mask_tensor'] = output_tensor['ray_mask']
        #
        output_sampled_dict['raypos_tensor'] = output_tensor['raypos_tensor']
        output_sampled_dict['vsize_np'] = output_tensor['vsize']

        return output_sampled_dict

    def get_point_indices(self, ray_dirs_tensor, cam_rot_tensor, cam_pos_tensor, near_plane,
                          far_plane,
                          point_cloud):
        # setup
        cam_pos_tensor = cam_pos_tensor.squeeze(0).to(point_cloud.device)
        cam_rot_tensor = cam_rot_tensor.squeeze(0).to(point_cloud.device)  # c2w
        # ray_dirs_tensor = inputs["raydir"]
        ray_dirs_tensor = ray_dirs_tensor.squeeze(0).to(point_cloud.device)


        # point_xyz_pers_tensor = self.w2pers(point_cloud, cam_rot_tensor, cam_pos_tensor)
        actual_numpoints_tensor = torch.ones([1], device=point_cloud.device,
                                             dtype=torch.int32) * point_cloud.shape[1]
        # print("pixel_idx_tensor", pixel_idx_tensor)
        # print("point_xyz_pers_tensor", point_xyz_pers_tensor.shape)
        # print("actual_numpoints_tensor", actual_numpoints_tensor.shape)
        # sample_pidx_tensor: B, R, SR, K

        # print("ray_dirs_tensor", ray_dirs_tensor.shape, self.xyz.shape)

        sample_pidx_tensor, sample_loc_cam_coor_tensor, sample_loc_world_coor_tensor, sample_ray_dirs_tensor, \
        ray_mask_tensor, vsize, ranges, raypos_tensor = \
            self.querier.query_points(point_cloud,
                                      actual_numpoints_tensor, near_plane, far_plane, ray_dirs_tensor,
                                      cam_pos_tensor, cam_rot_tensor)

        # print("ray_mask_tensor",ray_mask_tensor.shape)
        # self.pers2img(point_xyz_pers_tensor, pixel_idx_tensor.cpu().numpy(), pixel_idx_cur_tensor.cpu().numpy(), ray_mask_tensor.cpu().numpy(), sample_pidx_tensor.cpu().numpy(), ranges, h, w, inputs)

        B, _, SR, K = sample_pidx_tensor.shape
        # if vox_query:
        #     if sample_pidx_tensor.shape[1] > 0:
        #         sample_pidx_tensor = self.query_vox_grid(sample_loc_world_coor_tensor, self.full_grid_idx, self.space_min, self.grid_vox_sz)
        #     else:
        #         sample_pidx_tensor = torch.zeros([B, 0, SR, 8], device=sample_pidx_tensor.device, dtype=sample_pidx_tensor.dtype)

        # Output dict
        output_tensor = dict()
        output_tensor['sample_pidx'] = sample_pidx_tensor
        output_tensor['sample_loc_cam_coor'] = sample_loc_cam_coor_tensor
        output_tensor['ray_mask'] = ray_mask_tensor
        # output_tensor['point_xyz_pers'] = point_xyz_pers_tensor
        output_tensor['sample_loc_w_coor'] = sample_loc_world_coor_tensor
        output_tensor['sample_ray_dirs'] = sample_ray_dirs_tensor
        output_tensor['vsize'] = vsize
        output_tensor['raypos_tensor'] = raypos_tensor

        return output_tensor

    def save_points(self):
        return torch.zeros((0, 1))


@POINTS_REGISTRY.register()
class NeuralPoints_Simple_Incre(nn.Module):
    def __init__(self, cfg, **kwargs, ):
        super().__init__()
        self.opt = cfg
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        device = torch.device("cuda:0")
        print("Device: ", self.device, " ", device)
        if 'log_path' in kwargs:
            self.log_path = kwargs['log_path']
        else:
            self.log_path = None

        # set parameters
        self.test_ = torch.rand((1,1,1), device=device, dtype=torch.float32)
        #
        self.ranges = torch.FloatTensor(strlist_to_floatlist(cfg.DATA.RANGES), )
        self.min_xyz = self.ranges[:3]
        self.voxel_size = torch.FloatTensor(strlist_to_floatlist(self.cfg.MODEL.NEURAL_POINTS.VOX_SIZE))

        kernel_size = strlist_to_floatlist(cfg.MODEL.NEURAL_POINTS.KERNEL_SIZE)
        self.query_size = kernel_size
        if cfg.MODEL.NEURAL_POINTS.WCOOR_QUERY:
            self.lighting_fast_querier = lighting_fast_querier_w
        else:
            raise NotImplemented()
        # self.lighting_fast_querier = lighting_fast_querier_w if self.opt.wcoord_query > 0 else lighting_fast_querier_p
        self.querier = self.lighting_fast_querier(device, self.opt)
        self.reset_points()
        # Load the pretrained_stage1 checkpoints of latest points
        # if self.cfg.MODEL.NEURAL_POINTS.LOAD_POINTS:
        #     self.load_points()

    def reset_points(self):
        # if hasattr(self, 'points_pos') and self.log_path is not None and (self.points_pos.size(0) > 0):
        #     # save points
        #     saved_name = "latest_points.pkl"
        #     saved_dict = {
        #         'points_pos': self.points_pos,
        #         'points_emb': self.points_emb,
        #         'points_color': self.points_color,
        #         'points_dir': self.points_dir,
        #     }
        #     torch.save(saved_dict, os.path.join(self.log_path, saved_name))
        # register default points
        self.register_buffer('points_pos', torch.zeros((0, 3)).cuda())
        self.register_buffer('points_emb', torch.zeros(0, 32).cuda())
        self.register_buffer('points_color', torch.zeros((0, 3)).cuda())
        self.register_buffer('points_dir', torch.zeros((0, 3)).cuda())

    def save_points(self):
        saved_dict = {
            'points_pos': self.points_pos,
            'points_emb': self.points_emb,
            'points_color': self.points_color,
            'points_dir': self.points_dir,
        }
        return saved_dict

    def load_points(self,  points):
        # if self.cfg.MODEL.NEURAL_POINTS.LOAD_POINTS_PATH is not None:
        #     points = torch.load(self.cfg.MODEL.NEURAL_POINTS.LOAD_POINTS_PATH + "latest_points.pkl")
        self.register_buffer('points_pos', points['points_pos'].cuda())
        self.register_buffer('points_emb', points['points_emb'].cuda())
        self.register_buffer('points_color', points['points_color'].cuda())
        self.register_buffer('points_dir', points['points_dir'].cuda())

    def register_points(self, points_pos, points_emb, points_color, points_dir):
        # All type: N x D
        self.register_buffer('points_pos', points_pos.detach())
        self.register_buffer('points_emb', points_emb.detach())
        self.register_buffer('points_color', points_color.detach())
        self.register_buffer('points_dir', points_dir.detach())


    def incre_points(self, incre_pos, incre_emb, base_min, voxel_size, incre_color, incre_dir):
        if incre_pos is None and incre_emb is None and incre_color is None:
            return self.points_pos, self.points_emb, self.points_color, self.points_dir
        combined_pos, combined_emb, min_idx = incre_points(self.points_pos, incre_pos, base_min, voxel_size,
                                                           self.points_emb,
                                                           incre_emb)
        points_color_concat = torch.cat((self.points_color, incre_color), dim=0)
        c_points_color = torch.index_select(points_color_concat, dim=0, index=min_idx)
        points_dir_concat = torch.cat((self.points_dir, incre_dir), dim=0)
        c_points_dir = torch.index_select(points_dir_concat, dim=0, index=min_idx)

        return combined_pos, combined_emb, c_points_color, c_points_dir

    def forward(self, inputs, **kwargs):
        '''

        :param inputs:
        :return:
        '''
        """ 
        First prepare the input for this func!
        """
        if inputs['reset_points']:
            self.reset_points()
        # setup for points input
        points_info = inputs['points_info']
        # Incremental points information, such as, position and embeddings
        point_cloud_pos = points_info['points_position']
        points_embeddings = points_info['points_embeddings']
        points_color = points_info['points_color']
        points_dir = points_info['points_dir']
        # setup for raydir
        ray_dir_tensor = inputs["raydir"]
        # Setup for other params
        camrotc2w, campos, near_plane, far_plane, intrinsic = inputs["camrotc2w"], inputs["campos"], \
                                                              inputs["near"], inputs["far"], \
                                                              inputs["intrinsic"]
        """
        Do the computation!
        """
        # concat the incremental points with current points
        combined_pos, combined_emb, c_points_color, c_points_dir = self.incre_points(point_cloud_pos,
                                                                                     points_embeddings,
                                                                                     self.min_xyz.to(self.device),
                                                                                     self.voxel_size.to(self.device),
                                                                                     points_color,
                                                                                     points_dir)

        # The bug: sample_pidx is size [:, 0, ...]
        # pdb.set_trace()
        if len(combined_pos.size()) == 2:
            combined_pos = combined_pos.unsqueeze(0)
        output_tensor = self.get_point_indices(
            ray_dir_tensor, camrotc2w, campos, torch.min(near_plane).cpu().numpy(),
            torch.max(far_plane).cpu().numpy(),
            combined_pos)  # default opt.NN = 2, so vox_query is false

        # output_tensor['sample_pidx'] = sample_pidx_tensor
        #         output_tensor['sample_loc_cam_coor'] = sample_loc_cam_coor_tensor
        #         output_tensor['ray_mask'] = ray_mask_tensor
        #         output_tensor['point_xyz_pers'] = point_xyz_pers_tensor
        #         output_tensor['sample_loc_w_coor'] = sample_loc_world_coor_tensor
        #         output_tensor['sample_ray_dirs'] = sample_ray_dirs_tensor
        #         output_tensor['visze'] = vsize
        # obtain target params from output_tensor
        sample_pidx = output_tensor['sample_pidx']
        # point_xyz_pers_tensor = output_tensor['point_xyz_pers']

        sample_pnt_mask = sample_pidx >= 0
        B, R, SR, K = sample_pidx.shape
        # B: batch, R: ray number, K: max neural points each group,
        # SR: max shading points number each ray
        # if R == 0:
        #     pdb.set_trace()
        sample_pidx = torch.clamp(sample_pidx, min=0).view(-1).long()

        # For sampled embedding
        # TODO: NOTE!!! I emit the position in the camera coordinate!
        sampled_embedding_o = torch.index_select(
            torch.cat([combined_pos, combined_emb[None, ...]], dim=-1), 1, sample_pidx)
        sampled_embedding = sampled_embedding_o.view(B, R, SR, K,
                                                     combined_emb.shape[-1] + combined_pos.shape[-1])
        # For colors
        sampled_color = torch.index_select(c_points_color, 0, sample_pidx).view(B, R, SR, K, c_points_color.shape[-1])
        # For dir
        sampled_dir = torch.index_select(c_points_dir, 0, sample_pidx).view(B, R, SR, K, c_points_dir.shape[-1])
        # For conf
        sampled_conf = None

        """ Put the value in the output dict"""
        output_sampled_dict = dict()
        output_sampled_dict['sampled_color'] = sampled_color
        # output_sampled_dict['sampled_Rw2c'] = sampled_Rw2c
        output_sampled_dict['sampled_dir'] = sampled_dir
        output_sampled_dict['sampled_conf'] = sampled_conf
        output_sampled_dict['sampled_embedding'] = sampled_embedding[..., 3:]
        # output_sampled_dict['sampled_xyz_pers'] = sampled_embedding[..., 3:6]
        output_sampled_dict['sampled_xyz'] = sampled_embedding[..., :3]
        output_sampled_dict['sampled_pnt_mask'] = sample_pnt_mask
        output_sampled_dict['sample_loc_cam_coor'] = output_tensor['sample_loc_cam_coor']
        output_sampled_dict['sample_loc_w_coor'] = output_tensor['sample_loc_w_coor']
        output_sampled_dict['sample_ray_dirs'] = output_tensor['sample_ray_dirs']
        output_sampled_dict['ray_mask_tensor'] = output_tensor['ray_mask']
        #
        output_sampled_dict['raypos_tensor'] = output_tensor['raypos_tensor']
        output_sampled_dict['vsize_np'] = output_tensor['vsize']

        if inputs['register_flag']:
            self.register_points(combined_pos, combined_emb, c_points_color, c_points_dir)

        return output_sampled_dict

    def get_point_indices(self, ray_dirs_tensor, cam_rot_tensor, cam_pos_tensor, near_plane, far_plane, point_cloud):
        # setup
        cam_pos_tensor = cam_pos_tensor.squeeze(0).to(point_cloud.device)
        cam_rot_tensor = cam_rot_tensor.squeeze(0).to(point_cloud.device)  # c2w
        ray_dirs_tensor = ray_dirs_tensor.squeeze(0).to(point_cloud.device)

        actual_numpoints_tensor = torch.ones([1], device=point_cloud.device,
                                             dtype=torch.int32) * point_cloud.shape[1]

        sample_pidx_tensor, sample_loc_cam_coor_tensor, sample_loc_world_coor_tensor, sample_ray_dirs_tensor, \
        ray_mask_tensor, vsize, ranges, raypos_tensor = \
            self.querier.query_points(point_cloud,
                                      actual_numpoints_tensor, near_plane, far_plane, ray_dirs_tensor,
                                      cam_pos_tensor, cam_rot_tensor)
        # Output dict
        output_tensor = dict()
        output_tensor['sample_pidx'] = sample_pidx_tensor
        output_tensor['sample_loc_cam_coor'] = sample_loc_cam_coor_tensor
        output_tensor['ray_mask'] = ray_mask_tensor
        output_tensor['sample_loc_w_coor'] = sample_loc_world_coor_tensor
        output_tensor['sample_ray_dirs'] = sample_ray_dirs_tensor
        output_tensor['vsize'] = vsize
        output_tensor['raypos_tensor'] = raypos_tensor
        return output_tensor

    def forward_render_only(self, inputs, **kwargs):
        # setup
        # setup for raydir
        ray_dir_tensor = inputs["raydir"]
        # Setup for other params
        camrotc2w, campos, near_plane, far_plane, intrinsic = inputs["camrotc2w"], inputs["campos"], \
                                                              inputs["near"], inputs["far"], \
                                                              inputs["intrinsic"]
        combined_pos = self.points_pos
        combined_emb = self.points_emb
        c_points_color = self.points_color
        c_points_dir = self.points_dir
        #
        if len(combined_pos.size()) == 2:
            combined_pos = combined_pos.unsqueeze(0)
        output_tensor = self.get_point_indices(
            ray_dir_tensor, camrotc2w, campos, torch.min(near_plane).cpu().numpy(),
            torch.max(far_plane).cpu().numpy(),
            combined_pos)  # default opt.NN = 2, so vox_query is false

        # output_tensor['sample_pidx'] = sample_pidx_tensor
        #         output_tensor['sample_loc_cam_coor'] = sample_loc_cam_coor_tensor
        #         output_tensor['ray_mask'] = ray_mask_tensor
        #         output_tensor['point_xyz_pers'] = point_xyz_pers_tensor
        #         output_tensor['sample_loc_w_coor'] = sample_loc_world_coor_tensor
        #         output_tensor['sample_ray_dirs'] = sample_ray_dirs_tensor
        #         output_tensor['visze'] = vsize
        # obtain target params from output_tensor
        sample_pidx = output_tensor['sample_pidx']
        # point_xyz_pers_tensor = output_tensor['point_xyz_pers']

        sample_pnt_mask = sample_pidx >= 0
        B, R, SR, K = sample_pidx.shape
        # B: batch, R: ray number, K: max neural points each group,
        # SR: max shading points number each ray
        # if R == 0:
        #     pdb.set_trace()
        sample_pidx = torch.clamp(sample_pidx, min=0).view(-1).long()

        # For sampled embedding
        # TODO: NOTE!!! I emit the position in the camera coordinate!
        sampled_embedding_o = torch.index_select(
            torch.cat([combined_pos, combined_emb[None, ...]], dim=-1), 1, sample_pidx)
        sampled_embedding = sampled_embedding_o.view(B, R, SR, K,
                                                     combined_emb.shape[-1] + combined_pos.shape[-1])
        # For colors
        sampled_color = torch.index_select(c_points_color, 0, sample_pidx).view(B, R, SR, K, c_points_color.shape[-1])
        # For dir
        sampled_dir = torch.index_select(c_points_dir, 0, sample_pidx).view(B, R, SR, K, c_points_dir.shape[-1])
        # For conf
        sampled_conf = None

        """ Put the value in the output dict"""
        output_sampled_dict = dict()
        output_sampled_dict['sampled_color'] = sampled_color
        # output_sampled_dict['sampled_Rw2c'] = sampled_Rw2c
        output_sampled_dict['sampled_dir'] = sampled_dir
        output_sampled_dict['sampled_conf'] = sampled_conf
        output_sampled_dict['sampled_embedding'] = sampled_embedding[..., 3:]
        # output_sampled_dict['sampled_xyz_pers'] = sampled_embedding[..., 3:6]
        output_sampled_dict['sampled_xyz'] = sampled_embedding[..., :3]
        output_sampled_dict['sampled_pnt_mask'] = sample_pnt_mask
        output_sampled_dict['sample_loc_cam_coor'] = output_tensor['sample_loc_cam_coor']
        output_sampled_dict['sample_loc_w_coor'] = output_tensor['sample_loc_w_coor']
        output_sampled_dict['sample_ray_dirs'] = output_tensor['sample_ray_dirs']
        output_sampled_dict['ray_mask_tensor'] = output_tensor['ray_mask']
        #
        output_sampled_dict['raypos_tensor'] = output_tensor['raypos_tensor']
        output_sampled_dict['vsize_np'] = output_tensor['vsize']


        return output_sampled_dict

    def forward_register_points(self, inputs, **kwargs):
        '''
        This func only register points from inputs.
        :param inputs:
        :return:
        '''
        """ 
        First prepare the input for this func!
        """
        if inputs['reset_points']:
            self.reset_points()
            print("!!! Reset Points !!!")
        # setup for points input
        points_info = inputs['points_info']
        # Incremental points information, such as, position and embeddings
        point_cloud_pos = points_info['points_position']
        points_embeddings = points_info['points_embeddings']
        points_color = points_info['points_color']
        points_dir = points_info['points_dir']
        """
        Do the computation!
        """
        # concat the incremental points with current points
        combined_pos, combined_emb, c_points_color, c_points_dir = self.incre_points(point_cloud_pos,
                                                                                     points_embeddings,
                                                                                     self.min_xyz.to(self.device),
                                                                                     self.voxel_size.to(self.device),
                                                                                     points_color,
                                                                                     points_dir)

        if len(combined_pos.size()) == 2:
            combined_pos = combined_pos.unsqueeze(0)
        if inputs['register_flag']:
            self.register_points(combined_pos, combined_emb, c_points_color, c_points_dir)