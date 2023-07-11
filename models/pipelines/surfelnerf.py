import os
import math
import pdb
import time

import torchvision.transforms as T
transform = T.ToPILImage()
import torch
import numpy as np
from models.pipelines.base_pointnerf import PointsBaseModel
from models.aggregators.build import build_aggregators
from models.diff_rendering.diff_render_func import find_blend_function, find_tone_map, find_render_function
from models.diff_rendering.diff_ray_marching import ray_march, alpha_ray_march
from models.rasterizer.build import build_rasterizer
from models.losses.render_losses import BaseRenderLosses
from models.pipelines.build import MODEL_REGISTRY
from models.point_con.surfel_construction_net import SurfelRecon
from models.diff_rendering.helper import volumetric_rendering, get_t_vals, fill_bg, fill_bg_RGBGT
from pytorch3d.renderer import FoVPerspectiveCameras
from models.gru.build import build_fusion_net
from ..helpers.networks import positional_encoding
#
from pytorch3d.renderer.cameras import look_at_view_transform
from data.data_utils import get_dtu_raydir

def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)

@MODEL_REGISTRY.register()
class PointsSurfelModel(PointsBaseModel):
    """
    This model is different from original class:
    1) We first combine the point to select valid points instead of obtaining all pixel points from points
       reconstruction network, which maybe can save the memory...
    2) The last part we follow PointNeRF.
    """
    def __init__(self, cfg, **kwargs):
        super(PointsSurfelModel, self).__init__(cfg=cfg, **kwargs)
        build_flag = kwargs.get('build', True)
        campos = np.array([6.1, 2.49, 3.])[None,]
        look_at_dir = np.array([0, 0, -1])[None]
        up = np.array([0,1,0])[None]
        # self.hard_code_camera = self._build_camera_object(self.cam_para_hardcode)
        # Some hard code for logger
        self.logger_iter = 0
        self.time_logger_dict = dict()
        self.start_time = time.time()
        self.surfel_num_dict = dict()
        #
        self.scene_name = None
        #
        if build_flag:
            ### init network modules
            # point aggregator for rendering
            self.aggregator = build_aggregators(cfg)

            # build the point reconstruction network for extracting features of each pixel
            self.point_recon_net = SurfelRecon(cfg)

            ### init loss func
            self.loss_func = BaseRenderLosses(cfg)

            # ## We use rasterizer to replace original neural points module.
            self.rasterizer = build_rasterizer(cfg)

            if cfg.MODEL.FUSION.BUILD_FUSION:
                ### We build the gru fusion
                self.fusion_net = build_fusion_net(cfg)

            ### init optimizer
            if self.cfg.IS_TRAIN:
                self.setup_optimizer()
                self.setup_scheduler()

    def to_gpu(self):
        self.aggregator.to(self.device)
        self.point_recon_net.to(self.device)
        self.loss_func.to(self.device)
        self.bg_color = self.bg_color.cuda()
        self.rasterizer.to(self.device)
        if self.cfg.MODEL.FUSION.BUILD_FUSION:
            self.fusion_net.to(self.device)

    def setup_global_points(self, global_points_info, reset_global_points=False, features=None, points=None,
                            points_mask=None):
        if points is None:
            points = global_points_info['points_pos']
        normals = global_points_info['normals']
        if features is None:
            features = global_points_info['features']
        weights = global_points_info['weights']
        if points_mask is None:
            points_mask = global_points_info['point_mask']
        colors = global_points_info['colors']
        dirs = global_points_info['pcd_dirs']
        if len(features.size()) == 2:
            features = features.unsqueeze(0)
        # update features via gru
        features_new = self.setup_global_points_gru_feats(features, weights)
        self.rasterizer.setup_global_model(points, normals, features_new, colors=colors, pcd_dirs=dirs, weights=weights,
                                           points_mask=points_mask, reset=reset_global_points)

    def setup_global_points_gru_feats(self, features, weights):
        return features

    def _rasterizer_points(self, add_points_info, target_camera):
        # obtain local points info
        local_points_mask = add_points_info['point_mask']
        local_depth = add_points_info['depth']
        # setup hyper-parameters
        depth_merge_thres = self.cfg.depth_merge_thres
        # conduct rasterization
        empty_local_index, merge_local_index, merge_global_index, first_global_point_idx, filtered_global_mask, unchanged_global_index = \
            self.compare_pixel(local_depth, target_camera, depth_merge_thres,
                               local_point_mask=local_points_mask)
        #
        return empty_local_index, merge_local_index, merge_global_index, first_global_point_idx, filtered_global_mask, unchanged_global_index

    def fuse_with_new_points(self, local_points_info, camera, new_image, image_wh):
        assert self.rasterizer.global_points is not None
        empty_local_index, merge_local_index, merge_global_index, first_global_point_idx, filtered_global_mask, unchanged_global_index = \
            self._rasterizer_points(local_points_info, camera)
        pcd_dirs = local_points_info['pcd_dirs']
        # bs = new_image.size(0)
        # pcd_dirs = pcd_dirs.reshape(bs, image_wh[0] * image_wh[1], 3)
        input_for_surfel_recon = {
            'input_image': new_image,
            'image_wh': image_wh,
            # 'points_pos': input['points_pos']
            'add_local_index': empty_local_index,
            'merge_local_index': merge_local_index,
            'pcd_dirs': pcd_dirs.to(self.device)
        }
        # extract local features and conduct data association for update
        add_local_feats, add_local_rgb, merge_local_feats, merge_local_rgb = self.point_recon_net.forward_w_index(
            input_for_surfel_recon)
        # Update the target points
        # We split the points features in the above, and obtain other info from below func.
        local_updated_dict = self.obtain_need_local_info(local_points_info, empty_local_index, merge_local_index,
                                                         add_local_feats, merge_local_feats)
        self.fuse_to_global(cand_local_points=local_updated_dict['cand_local_points'],
                            cand_local_normals=local_updated_dict['cand_local_normals'],
                            cand_local_features=local_updated_dict['cand_local_features'],
                            cand_local_weights=local_updated_dict['cand_local_weights'],
                            merge_global_index=merge_global_index,
                            filter_global_index=filtered_global_mask,
                            cand_local_dirs=local_updated_dict['cand_local_dirs'],
                            cand_local_colors=local_updated_dict['cand_local_colors'],
                            )
        #
        local_feats = local_updated_dict['add_local_features']
        update_local_feats = self.fuse_with_new_points_gru_feats(local_feats, local_updated_dict['add_local_weights'])
        #
        self.add_local_points(local_points=local_updated_dict['add_local_points'],
                              local_normals=local_updated_dict['add_local_normals'],
                              local_featurs=update_local_feats,
                              local_weights=local_updated_dict['add_local_weights'],
                              local_colors=local_updated_dict['add_local_colors'],
                              local_dirs=local_updated_dict['add_local_dirs'],
                              )

    def fuse_with_new_points_gru_feats(self, local_feats, weights):
        # This func for GRU func, since incremental features need to be fed into the GRU module.
        # This is same as the process in the self.setup_global_points for new surfels.
        return local_feats

    def _select_points(self, mask):
        feats = self.rasterizer.global_features[:, mask]
        # feats = feats.unsqueeze(0)
        dirs = self.rasterizer.global_dirs[:, mask]
        # dirs = dirs.unsqueeze(0)
        colors = self.rasterizer.global_colors[:, mask]
        # colors = colors.unsqueeze(0)
        # pdb.set_trace()
        return feats, dirs, colors

    def rasterize_global_with_pixelcoord_ft(self, camera, sampled_pixel=None, return_all_pixel=False):
        output = self.rasterizer.rasterize(camera)
        mask = output[2]
        #
        feats, dirs, colors = self._select_points(mask)
        fragments = output[0]
        filtered_points = output[1]
        feats1 = filtered_points.features_packed()
        pdb.set_trace()

        valid_sampled_pixel = self.cfg.DATA.RANDOM_SAMPLE_SIZE
        #
        idx = fragments.idx  # BS x H x W x SR x K
        zbuf = fragments.zbuf
        qvalue = fragments.qvalue
        # pdb.set_trace()
        sr = idx.size(-2)
        k = idx.size(-1)
        # feats = filtered_points.features_packed()
        # feats = feats.unsqueeze(0)  # (1, N, num_feats)
        # dirs = filtered_points.dirs_packed()
        # dirs = dirs.unsqueeze(0)
        # colors = filtered_points.colors_packed()
        # colors = colors.unsqueeze(0)
        # pdb.set_trace()
        if return_all_pixel:
            sampled_idx = idx[0]
            sampled_zbuf = zbuf[0]  # Hs x Ws x SR x K
            sampled_qvalue = qvalue[0]  # Hs x Ws x SR x K
            #
            # pdb.set_trace()
            if self.cfg.MODEL.RASTERIZER.FILTER_NEAR_POINTS:
                sampled_idx = torch.where(sampled_zbuf < 0.1, torch.ones_like(sampled_zbuf, dtype=torch.int) * -1, sampled_idx)
            #
            # return target
            dis_point_to_ray = self._estimate_real_dis(sampled_zbuf, sampled_qvalue,
                                                       self.cfg.DATA.NEAR_PLANE)  # Hs x Ws x SR x K
            idx_list = sampled_idx.reshape(-1)  # Hs*Ws*SR*K
            # obtain valid ray mask
            sampled_valid_ray_mask = sampled_idx > 0  # (Hs, Ws, SR, K)
            idx_list_clamp = torch.clamp(idx_list, min=0)
            idx_list_clamp = idx_list_clamp.long()
            sampled_feats = torch.index_select(feats, index=idx_list_clamp, dim=1)
            sampled_feats = sampled_feats.reshape(1, sampled_idx.size(0) * sampled_idx.size(1), sr, k, -1)
            sampled_dis_to_ray = dis_point_to_ray.reshape(1, sampled_idx.size(0) * sampled_idx.size(1), sr, k, -1)
            sampled_dirs = torch.index_select(dirs, index=idx_list_clamp, dim=1)
            sampled_dirs = sampled_dirs.reshape(1, sampled_idx.size(0) * sampled_idx.size(1), sr, k, 3)
            sampled_colors = torch.index_select(colors, index=idx_list_clamp, dim=1)
            sampled_colors = sampled_colors.reshape(1, sampled_idx.size(0) * sampled_idx.size(1), sr, k, 3)
            return sampled_feats, sampled_dis_to_ray, sampled_zbuf, sampled_dirs, sampled_colors, sampled_valid_ray_mask, sampled_idx

        # valid pixel
        px, py = sampled_pixel
        sampled_idx = idx[0, py, px]  # Hs x Ws x SR x K
        sampled_zbuf = zbuf[0, py, px]  # Hs x Ws x SR x K
        sampled_qvalue = qvalue[0, py, px]  # Hs x Ws x SR x K
        #
        if self.cfg.MODEL.RASTERIZER.FILTER_NEAR_POINTS:
            sampled_idx = torch.where(sampled_zbuf < 0.25, torch.ones_like(sampled_zbuf, dtype=torch.int) * -1, sampled_idx)
        # return target
        dis_point_to_ray = self._estimate_real_dis(sampled_zbuf, sampled_qvalue, self.cfg.DATA.NEAR_PLANE)  # Hs x Ws x SR x K
        idx_list = sampled_idx.reshape(-1)  # Hs*Ws*SR*K
        # obtain valid ray mask
        sampled_valid_ray_mask = sampled_idx > 0  # (Hs, Ws, SR, K)
        idx_list_clamp = torch.clamp(idx_list, min=0)
        idx_list_clamp = idx_list_clamp.long()
        sampled_feats = torch.index_select(feats, index=idx_list_clamp, dim=1)
        sampled_feats = sampled_feats.reshape(1, valid_sampled_pixel * valid_sampled_pixel, sr, k, -1)
        sampled_dis_to_ray = dis_point_to_ray.reshape(1, valid_sampled_pixel * valid_sampled_pixel, sr, k, -1)
        sampled_dirs = torch.index_select(dirs, index=idx_list_clamp, dim=1)
        sampled_dirs = sampled_dirs.reshape(1, valid_sampled_pixel * valid_sampled_pixel, sr, k, 3)
        sampled_colors = torch.index_select(colors, index=idx_list_clamp, dim=1)
        sampled_colors = sampled_colors.reshape(1, valid_sampled_pixel * valid_sampled_pixel, sr, k, 3)
        return sampled_feats, sampled_dis_to_ray, sampled_zbuf, sampled_dirs, sampled_colors, sampled_valid_ray_mask, sampled_idx

    def rasterize_global_with_pixelcoord(self, camera, sampled_pixel=None, return_all_pixel=False):
        output = self.rasterizer.rasterize(camera)
        #
        fragments = output[0]
        filtered_points = output[1]
        valid_sampled_pixel = self.cfg.DATA.RANDOM_SAMPLE_SIZE
        #
        idx = fragments.idx  # BS x H x W x SR x K
        zbuf = fragments.zbuf
        qvalue = fragments.qvalue
        dx = fragments.dx
        dy = fragments.dy
        rx = fragments.rx
        ry = fragments.ry
        #
        sr = idx.size(-2)
        k = idx.size(-1)
        feats = filtered_points.features_packed()
        feats = feats.unsqueeze(0)  # (1, N, num_feats)
        dirs = filtered_points.dirs_packed()
        dirs = dirs.unsqueeze(0)
        colors = filtered_points.colors_packed()
        colors = colors.unsqueeze(0)
        #
        # pdb.set_trace()
        if return_all_pixel:
            sampled_idx = idx[0]
            sampled_zbuf = zbuf[0]  # Hs x Ws x SR x K
            sampled_qvalue = qvalue[0]  # Hs x Ws x SR x K
            #
            if self.cfg.MODEL.RASTERIZER.FILTER_NEAR_POINTS:
                sampled_idx = torch.where(sampled_zbuf < 0.1, torch.ones_like(sampled_zbuf, dtype=torch.int) * -1, sampled_idx)
            # return target
            dis_point_to_ray = self._estimate_real_dis(sampled_zbuf, sampled_qvalue,
                                                       self.cfg.DATA.NEAR_PLANE)  # Hs x Ws x SR x K
            idx_list = sampled_idx.reshape(-1)  # Hs*Ws*SR*K
            # obtain valid ray mask
            sampled_valid_ray_mask = sampled_idx > 0  # (Hs, Ws, SR, K)
            idx_list_clamp = torch.clamp(idx_list, min=0)
            idx_list_clamp = idx_list_clamp.long()
            sampled_feats = torch.index_select(feats, index=idx_list_clamp, dim=1)
            sampled_feats = sampled_feats.reshape(1, sampled_idx.size(0) * sampled_idx.size(1), sr, k, -1)
            sampled_dis_to_ray = dis_point_to_ray.reshape(1, sampled_idx.size(0) * sampled_idx.size(1), sr, k, -1)
            sampled_dirs = torch.index_select(dirs, index=idx_list_clamp, dim=1)
            sampled_dirs = sampled_dirs.reshape(1, sampled_idx.size(0) * sampled_idx.size(1), sr, k, 3)
            sampled_colors = torch.index_select(colors, index=idx_list_clamp, dim=1)
            sampled_colors = sampled_colors.reshape(1, sampled_idx.size(0) * sampled_idx.size(1), sr, k, 3)
            #
            pos_x = dx/rx
            pos_y = dy/ry
            pos_x = pos_x.reshape(1, -1, 1)
            pos_y = pos_y.reshape(1, -1, 1)
            sampled_pos_x = torch.index_select(pos_x, index=idx_list_clamp, dim=1)
            sampled_pos_x = sampled_pos_x.reshape(1, sampled_idx.size(0) * sampled_idx.size(1), sr, k, 1)
            sampled_pos_y = torch.index_select(pos_y, index=idx_list_clamp, dim=1)
            sampled_pos_y = sampled_pos_y.reshape(1, sampled_idx.size(0) * sampled_idx.size(1), sr, k, 1)
            return sampled_feats, sampled_dis_to_ray, sampled_zbuf, sampled_dirs, sampled_colors, \
                   sampled_valid_ray_mask, sampled_idx, sampled_pos_x, sampled_pos_y

        # valid pixel
        px, py = sampled_pixel
        sampled_idx = idx[0, py, px]  # Hs x Ws x SR x K
        sampled_zbuf = zbuf[0, py, px]  # Hs x Ws x SR x K
        sampled_qvalue = qvalue[0, py, px]  # Hs x Ws x SR x K
        sampled_dx = dx[0, py, px]  # Hs x Ws x SR x K
        sampled_dy = dy[0, py, px]
        sampled_rx = rx[0, py, px]
        sampled_ry = ry[0, py, px]
        sampled_pos_x = sampled_dx / sampled_rx   # Hs x Ws x SR x K
        sampled_pos_y = sampled_dy / sampled_ry
        if self.cfg.MODEL.RASTERIZER.FILTER_NEAR_POINTS:
            sampled_idx = torch.where(sampled_zbuf < 0.25, torch.ones_like(sampled_zbuf, dtype=torch.int) * -1, sampled_idx)
        # return target
        dis_point_to_ray = self._estimate_real_dis(sampled_zbuf, sampled_qvalue, self.cfg.DATA.NEAR_PLANE)  # Hs x Ws x SR x K
        idx_list = sampled_idx.reshape(-1)  # Hs*Ws*SR*K
        # obtain valid ray mask
        sampled_valid_ray_mask = sampled_idx > 0  # (Hs, Ws, SR, K)
        idx_list_clamp = torch.clamp(idx_list, min=0)
        idx_list_clamp = idx_list_clamp.long()
        sampled_feats = torch.index_select(feats, index=idx_list_clamp, dim=1)
        sampled_feats = sampled_feats.reshape(1, valid_sampled_pixel * valid_sampled_pixel, sr, k, -1)
        sampled_dis_to_ray = dis_point_to_ray.reshape(1, valid_sampled_pixel * valid_sampled_pixel, sr, k, -1)
        sampled_dirs = torch.index_select(dirs, index=idx_list_clamp, dim=1)
        sampled_dirs = sampled_dirs.reshape(1, valid_sampled_pixel * valid_sampled_pixel, sr, k, 3)
        sampled_colors = torch.index_select(colors, index=idx_list_clamp, dim=1)
        sampled_colors = sampled_colors.reshape(1, valid_sampled_pixel * valid_sampled_pixel, sr, k, 3)
        sampled_pos_x = sampled_pos_x.reshape(1, valid_sampled_pixel * valid_sampled_pixel, sr, k, 1)
        sampled_pos_y = sampled_pos_y.reshape(1, valid_sampled_pixel * valid_sampled_pixel, sr, k, 1)
        return sampled_feats, sampled_dis_to_ray, sampled_zbuf, sampled_dirs, sampled_colors, \
               sampled_valid_ray_mask, sampled_idx, sampled_pos_x, sampled_pos_y

    def _estimate_real_dis(self, zbuf, qvalue, znear=0.1):
        scale_ = zbuf / znear
        real_dis = qvalue * scale_
        return real_dis

    def _render_one(self, camera, sampled_pixel, sampled_ray_dir):
        if self.cfg.TRAIN.FT_FLAG:
            sampled_feats, sampled_dis_to_ray, sampled_zbuf, sampled_pcd_dir, sampled_color, sampled_valid_ray_mask, sampled_idx = \
                self.rasterize_global_with_pixelcoord_ft(camera, sampled_pixel)
        else:
            sampled_feats, sampled_dis_to_ray, sampled_zbuf, sampled_pcd_dir, sampled_color, sampled_valid_ray_mask, sampled_idx, sampled_pos_x, sampled_pos_y = \
            self.rasterize_global_with_pixelcoord(camera, sampled_pixel)
        B, R, SR, K, _ = sampled_feats.size()
        # pdb.set_trace()
        # then we render this sampled pixels
        output = self.aggregator.forward(sampled_feats, sampled_color, sampled_dis_to_ray, sampled_pcd_dir,
                                         sampled_ray_dir,
                                         sampled_valid_ray_mask, sampled_pos_x, sampled_pos_y)
        # output = self._render_ray(sampled_feats, sampled_color, sampled_dis_to_ray, sampled_pcd_dir,
        #                           sampled_ray_dir, sampled_valid_ray_mask)
        # finally, we decode the output and obtain the color
        decoded_features, weights = output  # decoded_features: [B*R*K, 4]
        # pdb.set_trace()
        rgb = decoded_features[..., 1:]
        density = decoded_features[..., [0]]
        rgb = rgb.reshape(B, R, SR, 3)
        density = density.reshape(B, R, SR)

        ray_zbuf = sampled_zbuf[..., 0]  # B x Hs x Ws x SR
        ray_zbuf = ray_zbuf.reshape((sampled_zbuf.size(0), ray_zbuf.size(1) * ray_zbuf.size(2), ray_zbuf.size(3)))  # B x R x SR
        valid_pnt_mask = torch.any(sampled_valid_ray_mask, dim=-1).reshape(1, -1, SR)
        # valid_pnt_mask = sampled_valid_ray_mask.reshape(1, -1, K)  # B x R x K
        #
        dists = get_t_vals(depth_dist=ray_zbuf, valid_pnt_mask=valid_pnt_mask, far=self.cfg.DATA.FAR_PLANE)

        # rgb = decoded_features
        comp_rgb, acc, weights = volumetric_rendering(rgb, density, dists)
        # comp_rgb: (B, R, 3);
        ray_mask = torch.any(valid_pnt_mask, dim=-1)  # B x R
        rgb_color = fill_bg(ray_mask, comp_rgb, self.bg_color)
        #
        render_output = {
            'ray_mask': ray_mask, "coarse_color": rgb_color
        }
        # intermediate output
        sampled_output = {
            'colors': sampled_color, 'zbuf': sampled_zbuf, 'idx': sampled_idx
        }

        return render_output, sampled_output

    def _render_one_image(self, camera, sampled_pixel=None, sampled_ray_dir=None, rgb_gt=None):
        # First, we access the target camera
        # camera =
        # sampled_pixel = input['sampled_pixel']
        # sampled_pcd_dir = input['sampled_pcd_dir']
        # sampled_ray_dir = input['sampled_ray_dir']
        # valid_ray_mask = input['valid_ray_mask']

        # TODO: check here ...
        # sampled_feats, sampled_dis_to_ray, sampled_zbuf, sampled_pcd_dir, sampled_color, sampled_valid_ray_mask, sampled_idx = \
        #     self.rasterize_global_with_pixelcoord_ft(camera, sampled_pixel, return_all_pixel=True)
        sampled_feats, sampled_dis_to_ray, sampled_zbuf, sampled_pcd_dir, sampled_color, sampled_valid_ray_mask, sampled_idx, sampled_pos_x, sampled_pos_y = \
            self.rasterize_global_with_pixelcoord(camera, sampled_pixel, return_all_pixel=True)

        # below is same
        B, R, SR, K, _ = sampled_feats.size()  # R is h*w !
        # We split the R to render
        rgb_list = []
        density_list = []
        pixel_test = 200
        pixel_once_num = pixel_test*pixel_test  # Yes, hard code!
        render_times = math.ceil(R / pixel_once_num)
        sampled_valid_ray_mask_reshape = sampled_valid_ray_mask.reshape(B,R,SR,K)
        # pdb.set_trace()
        for index in range(render_times):
            begin_id = pixel_once_num * index
            end_id = min(pixel_once_num * (index + 1), R)
            sampled_feats_item = sampled_feats[:, begin_id:end_id, ]
            sampled_color_item = sampled_color[:, begin_id:end_id, ]
            sampled_dis_to_ray_item = sampled_dis_to_ray[:, begin_id:end_id, ]
            sampled_pcd_dir_item = sampled_pcd_dir[:, begin_id:end_id, ]
            sampled_ray_dir_item = sampled_ray_dir[:, begin_id:end_id, ]
            sampled_valid_ray_mask_item = sampled_valid_ray_mask_reshape[:, begin_id:end_id, ]
            sampled_valid_ray_mask_item = sampled_valid_ray_mask_item.reshape(B, end_id - begin_id,
                                                                              SR, K)
            sampled_pos_x_item = sampled_pos_x[:, begin_id:end_id, ]
            sampled_pos_y_item = sampled_pos_y[:, begin_id:end_id, ]
            output_item = self.aggregator.forward(sampled_feats_item, sampled_color_item, sampled_dis_to_ray_item,
                                                  sampled_pcd_dir_item, sampled_ray_dir_item,
                                                  sampled_valid_ray_mask_item, sampled_pos_x_item, sampled_pos_y_item
                                                  )
            # pdb.set_trace()
            decoded_features, weights = output_item  # decoded_features: [B*R*K, 4]
            rgb = decoded_features[..., 1:]
            density = decoded_features[..., [0]]
            rgb = rgb.reshape(B, end_id - begin_id, SR, 3)
            density = density.reshape(B, end_id - begin_id, SR)
            rgb_list.append(rgb)
            density_list.append(density)
        #
        rgb_final = torch.cat(rgb_list, dim=1)
        density_final = torch.cat(density_list, dim=1)

        # pdb.set_trace()
        ray_zbuf = sampled_zbuf[..., 0]  # Hs x Ws x SR
        ray_zbuf = ray_zbuf.reshape((1, ray_zbuf.size(0) * ray_zbuf.size(1), ray_zbuf.size(2)))  # B x R x SR
        valid_pnt_mask = torch.any(sampled_valid_ray_mask, dim=-1).reshape(1, -1, SR)
        # valid_pnt_mask = sampled_valid_ray_mask.reshape(1, -1, K)  # B x R x K
        #
        dists = get_t_vals(depth_dist=ray_zbuf, valid_pnt_mask=valid_pnt_mask, far=self.cfg.DATA.FAR_PLANE)

        # rgb = decoded_features
        comp_rgb, acc, weights = volumetric_rendering(rgb_final, density_final, dists)
        # comp_rgb: (B, R, 3);
        ray_mask = torch.any(valid_pnt_mask, dim=-1)  # B x R
        rgb_gt_reshape = rgb_gt.reshape(1, -1, 3)
        rgb_color = fill_bg_RGBGT(ray_mask, comp_rgb, self.bg_color, rgb_GT=rgb_gt_reshape, replace_GT_with_bg=True)
        #
        render_output = {
            'ray_mask': ray_mask, "coarse_color": rgb_color
        }
        # intermediate output
        sampled_output = {
            'colors': sampled_color, 'zbuf': sampled_zbuf, 'idx': sampled_idx
        }

        return render_output, sampled_output

    def obtain_need_local_info(self, local_points_info, add_local_index, merge_local_index, add_local_feats,
                               merge_local_feats):
        local_points = local_points_info['points_pos']
        local_normals = local_points_info['normals']
        # local_features = local_points_info['features']
        local_weights = local_points_info['weights']
        local_colors = local_points_info['colors']
        local_dirs = local_points_info['pcd_dirs']
        #
        add_local_points = local_points[0, add_local_index]
        add_local_normals = local_normals[0, add_local_index]
        add_local_features = add_local_feats.squeeze(0)
        add_local_weights = local_weights[0, add_local_index]
        add_local_colors = local_colors[0, add_local_index]
        add_local_dirs = local_dirs[0, add_local_index]
        #
        cand_local_points = local_points[0, merge_local_index]
        cand_local_normals = local_normals[0, merge_local_index]
        cand_local_features = merge_local_feats
        cand_local_colors = local_colors[0, merge_local_index]
        cand_local_dirs = local_dirs[0, merge_local_index]
        cand_local_weights = local_weights[0, merge_local_index]  # size(N)

        #
        local_updated_dict = {
            'add_local_points': add_local_points.unsqueeze(0),
            'add_local_normals': add_local_normals.unsqueeze(0),
            'add_local_features': add_local_features.unsqueeze(0),
            'add_local_weights': add_local_weights.unsqueeze(0),
            'add_local_colors': add_local_colors.unsqueeze(0),
            'add_local_dirs': add_local_dirs.unsqueeze(0),
            'cand_local_points': cand_local_points,
            'cand_local_normals': cand_local_normals,
            'cand_local_features': cand_local_features,
            'cand_local_weights': cand_local_weights,
            'cand_local_colors': cand_local_colors,
            'cand_local_dirs': cand_local_dirs,
        }
        return local_updated_dict

    def fuse_to_global(self,
               cand_local_points, cand_local_normals, cand_local_features, cand_local_weights,
               cand_local_colors, cand_local_dirs,
               merge_global_index, filter_global_index, **kwargs):
        # prepare points in the global model which need to be updated
        filter_g_points = self.rasterizer.global_points[0, filter_global_index]
        filter_g_normals = self.rasterizer.global_normals[0, filter_global_index]
        filter_g_features = self.rasterizer.global_features[0, filter_global_index]
        filter_g_weights = self.rasterizer.global_weights[0, filter_global_index]
        filter_g_colors = self.rasterizer.global_colors[0, filter_global_index]
        filter_g_dirs = self.rasterizer.global_dirs[0, filter_global_index]
        # prepare points that keep the original value
        # pdb.set_trace()
        unchanged_index = torch.ones_like(filter_global_index)
        nonzero = torch.where(filter_global_index)[0][merge_global_index]
        unchanged_index[nonzero] = 0

        cand_global_points = filter_g_points[merge_global_index]
        cand_global_normals = filter_g_normals[merge_global_index]
        cand_global_features = filter_g_features[merge_global_index]
        cand_global_colors = filter_g_colors[merge_global_index]
        cand_global_dirs = filter_g_dirs[merge_global_index]
        cand_global_weights = filter_g_weights[merge_global_index]
        # cand_global_weights = cand_global_weights
        # cand_local_weights = cand_local_weights
        # pdb.set_trace()
        # update corresponding points
        fused_points = (cand_global_weights * cand_global_points + cand_local_weights * cand_local_points) / (
                    cand_global_weights + cand_local_weights)
        fused_normals = (cand_global_weights * cand_global_normals + cand_local_weights * cand_local_normals) / (
                    cand_global_weights + cand_local_weights)
        # fused_features = (cand_global_weights * cand_global_features + cand_local_weights * cand_local_features) / (
        #             cand_global_weights + cand_local_weights)
        fused_colors = (cand_global_weights * cand_global_colors + cand_local_weights * cand_local_colors) / (
                cand_global_weights + cand_local_weights)
        fused_dirs = (cand_global_weights * cand_global_dirs + cand_local_weights * cand_local_dirs) / (
                cand_global_weights + cand_local_weights)
        fused_weights = (cand_global_weights + cand_local_weights)
        #
        filter_g_points[merge_global_index] = fused_points
        filter_g_normals[merge_global_index] = fused_normals
        # filter_g_features[merge_global_index] = fused_features
        filter_g_colors[merge_global_index] = fused_colors
        filter_g_dirs[merge_global_index] = fused_dirs
        filter_g_weights[merge_global_index] = fused_weights
        #
        # pdb.set_trace()
        self.rasterizer.global_points[0, filter_global_index] = filter_g_points
        self.rasterizer.global_normals[0, filter_global_index] = filter_g_normals
        # self.rasterizer.global_features[0, filter_global_index] = filter_g_features
        self.rasterizer.global_weights[0, filter_global_index] = filter_g_weights
        self.rasterizer.global_colors[0, filter_global_index] = filter_g_colors
        self.rasterizer.global_dirs[0, filter_global_index] = filter_g_dirs
        #
        self.fuse_to_global_feats(cand_local_features, cand_local_weights,
                             cand_global_weights, cand_global_features,
                             filter_g_features, merge_global_index, filter_global_index, cand_local_dirs,
                             cand_global_dirs)

    def fuse_to_global_feats(self, cand_local_features, cand_local_weights,
                             cand_global_weights, cand_global_features,
                             filter_g_features, merge_global_index, filter_global_index, cand_local_dirs,
                             cand_global_dirs):
        fused_features = (cand_global_weights * cand_global_features + cand_local_weights * cand_local_features) / (
                cand_global_weights + cand_local_weights)
        filter_g_features[merge_global_index] = fused_features
        self.rasterizer.global_features[0, filter_global_index] = filter_g_features

    def compare_pixel(self, local_depth, target_camera, depth_merge_thres, local_point_mask=None):
        if local_point_mask is not None:
            local_point_mask = local_point_mask.bool()
            assert local_point_mask.dtype == torch.bool
        # global_idx, global_zbuf, filter_points, filter_mask = self.rasterizer.rasterize(target_camera)
        output = self.rasterizer.rasterize(target_camera)
        fragments = output[0]
        # pdb.set_trace()

        global_idx = fragments.idx
        global_zbuf = fragments.zbuf
        # save
        self.global_idx = global_idx
        filter_mask = output[2]  # filter_mask is the mask that is used to filter the invalid point cloud.
        # pdb.set_trace()
        # check dim
        if len(global_idx.size()) == 5:
            global_idx = global_idx[..., 0]
            global_zbuf = global_zbuf[..., 0]
        idx_item = global_idx[:, :, :, :self.cfg.MODEL.NEURAL_POINTS.TOPK_RASTERIZED_SURFELS]  # extract the first index
        zbuf_item = global_zbuf[:, :, :, :self.cfg.MODEL.NEURAL_POINTS.TOPK_RASTERIZED_SURFELS]
        zbuf_item = zbuf_item.squeeze(-1)
        idx_item = idx_item.squeeze(-1)
        # pdb.set_trace()
        # check pixel whether need to be merged
        depth_sub = torch.abs(local_depth - zbuf_item)
        # pdb.set_trace()
        empty_flag = idx_item <= 0
        merge_flag = (depth_sub < depth_merge_thres)  # TODO: here maybe wrong since it can merge with the second later index instead of first only!
        # empty_flag = empty_flag.reshape(-1)

        if local_point_mask is not None:
            # pdb.set_trace()
            local_point_mask = local_point_mask.reshape(empty_flag.size())
            empty_flag[~local_point_mask] = torch.BoolTensor([False]).to(self.device)
            merge_flag[~local_point_mask] = torch.BoolTensor([False]).to(self.device)

        # pdb.set_trace()
        empty_local_index = torch.nonzero(empty_flag.reshape(-1))[:, 0]  # Size of: N x 1 -> N
        # need_update_flag = (merge_flag or empty_flag)
        # extract target idx of global model
        temp_merge_global_idx = idx_item[merge_flag]
        # temp_saved_global_idx = idx_item[~merge_flag]
        # pdb.set_trace()
        merge_global_index = temp_merge_global_idx.reshape(-1)  # N
        # unchanged_global_index = temp_saved_global_idx.reshape(-1)
        merge_local_index = torch.nonzero(merge_flag.reshape(-1))[:, 0]  # N x 1 -> N
        #
        idx_item_ = idx_item.reshape(-1)

        return empty_local_index.long(), merge_local_index.long(), merge_global_index.long(), idx_item_.long(), filter_mask, None

    def add_local_points(self, local_points, local_normals, local_featurs, local_weights,
                         local_colors=None, local_dirs=None):
        self.rasterizer.update_global_points(local_points, local_normals, local_featurs, local_weights,
                                             local_colors, local_dirs)

    #  -----------------------------------------------------------------------------------
    # Forward Modules
    # ------------------------------------------------------------------------------------
    def subforward_fusion_points_module(self, incre_input, img_wh, **kwargs):
        target_camera = self._build_camera_object(incre_input['camera'])
        self.fuse_with_new_points(incre_input['data'], target_camera, incre_input['rgb'], img_wh)

    def subforward_recon_surfel_module(self, input_data, img_wh, reset_global_points):
        surfel_input = {
            'input_image': input_data['rgb'],
            'image_wh': img_wh,
            'pcd_dirs': input_data['data']['pcd_dirs']
        }
        global_points_feats, _ = self.point_recon_net.obtain_points_from_img(surfel_input)
        self.setup_global_points(input_data['data'], features=global_points_feats,
                                 reset_global_points=reset_global_points)

    def subforward_render_module(self, target_data):
        target_camera = self._build_camera_object(target_data['camera'])
        sampled_ray_dir = target_data['sampled_ray']
        sampled_pixel = target_data['sampled_pidx']
        h, w, _ = sampled_ray_dir.size()
        sampled_ray_dir = sampled_ray_dir.reshape(1, h * w, 1, 3)
        rendered_output, sampled_output = self._render_one(target_camera, sampled_pixel, sampled_ray_dir)
        return rendered_output, sampled_output

    #  -----------------------------------------------------------------------------------
    # Forward code
    # ------------------------------------------------------------------------------------
    def forward_train(self, input, **kwargs):
        # pdb.set_trace()
        # get key params
        reset_global_points = input.pop('reset_global_points', True)
        need_point_con = input.pop('need_point_recon', True)
        if 'need_point_con' in kwargs:
            need_point_con = kwargs['need_point_con']
        input_data, incre_data, target_data = self._setup_input(input)

        img_wh = self.cfg.DATA.IMG_WH

        # For surfel reconstruction
        if not self.cfg.MODEL.NEED_POINTCON:
            need_point_con = False
        if 'need_point_con' in kwargs and kwargs['need_point_con'] is False:
            need_point_con = False

        if self.cfg.MODEL.FUSION.USE_GRU_DDP:
            if need_point_con:
                stage = "new input data"
            else:
                stage = "INCRE"
            try:
                print("Device {} : ({}) Scene: {}  ID: {} ".format(incre_data['rgb'].device, stage, incre_data['scene'], incre_data['data']['id'], ))
            except:
                print("Device {} : ({}) Scene: {}  ID: {} ".format( input_data['rgb'].device, stage, input_data['scene'],
                                                              input_data['data']['id'], ))
        # --------------------- For surfel reconstruction ----------------------------------
        if need_point_con:
            # Run surfel construction network to build the surfel.
            self.subforward_recon_surfel_module(input_data, img_wh, reset_global_points)

        # --------------------- For fusing points ----------------------------------
        if incre_data is None:
            incre_data = input_data
        else:
            self.subforward_fusion_points_module(incre_data, img_wh)

        # --------------------- For render --------------------------------------
        if target_data is None:
            raise RuntimeError("Third input is NONE!")
        rendered_output, sampled_output = self.subforward_render_module(target_data)

        # -------------------- Computing loss ----------------------------------
        losses_dict = self.loss_func.compute_color_l2loss(rendered_output, target_data['gt'])
        losses_total, log_vars = self._parse_losses(losses_dict)
        if 'no_grad' in kwargs:
            pass
        else:
            # optimize
            self._setup_zero_grad()
            if losses_total.data.item() == 0:
                # pdb.set_trace()
                print("There is no valid ray!!!")
            elif torch.isnan(losses_total):
                print("LOSS ERROR!")
                pdb.set_trace()

            else:
                try:
                    # pdb.set_trace()
                    # for gru
                    if self.cfg.MODEL.FUSION.BUILD_FUSION:
                        losses_total.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    else:
                        losses_total.backward()
                except:
                    print("one", incre_data['data']['scene'], incre_data['data']['id'], )
                    print("two", target_data['data']['scene'], target_data['data']['id'], )
                    print("LOSS ERROR!")
                    losses_total.backward()
                    pdb.set_trace()
            self._step_grad()
        output_final = dict()
        output_final['log_vars'] = log_vars
        # output_final['gt_image'] = second_data['rgb']
        output_final['sampled_output'] = sampled_output
        output_final['rendered_output'] = rendered_output
        # output_final['gt_pixel'] = incre_data['gt']
        return output_final

    def _tensor_to_img(self, tensor, name, index=0):
        t = tensor.cpu()
        t = t.reshape((256, 256, 3))
        t = t.permute((2, 0, 1))
        img = transform(t)
        # pdb.set_trace()
        new_name = os.path.join(self.cfg.given_path,
                                "{}_{}_{}_{}_id{}".format(self.cfg.DATA.SCENE_NAME, self.cfg.given_id,self.cfg.indx, self.cfg.indy,
                                                          int((19+float(index))//20))
                                )
        img.save(new_name+'.png')

    def forward_per_scene_opt(self, input, **kwargs):
        iter_now = kwargs.get('iterations', 0)
        input_data, _, target_data = self._setup_input(input)
        # --------------------- For render --------------------------------------
        if target_data is None:
            target_data = input_data
        rendered_output, sampled_output = self.subforward_render_module(target_data)

        # -------------------- Computing loss ----------------------------------
        losses_dict = self.loss_func.compute_color_l2loss(rendered_output, target_data['gt'])
        if self.cfg.save_path:
            self._tensor_to_img(rendered_output['coarse_color'], self.cfg.temp_name, index=iter_now)
        losses_total, log_vars = self._parse_losses(losses_dict)
        if 'no_grad' in kwargs:
            pass
        else:
            # optimize
            self._setup_zero_grad()
            if losses_total.data.item() == 0:
                # pdb.set_trace()
                print("There is no valid ray!!!")
            else:
                try:
                    losses_total.backward()
                except:
                    print("Losses are wrong!!!")
                    # pdb.set_trace()
            # pdb.set_trace()
            self._step_grad()
        output_final = dict()
        output_final['log_vars'] = log_vars
        # output_final['gt_image'] = second_data['rgb']
        output_final['sampled_output'] = sampled_output
        output_final['rendered_output'] = rendered_output
        # output_final['gt_pixel'] = input_data['gt']
        return output_final

    def forward_fuse(self, input, **kwargs):
        input_data, incre_data, _ = self._setup_input(input)
        img_wh = self.cfg.DATA.IMG_WH

        # For surfel reconstruction
        need_point_con = True
        if not self.cfg.MODEL.NEED_POINTCON:
            need_point_con = False
        if 'need_pointcon' in kwargs and kwargs['need_pointcon'] is False:
            need_point_con = False
        #
        if need_point_con:
            # Run surfel construction network to build the surfel.
            self.subforward_recon_surfel_module(input_data, img_wh, reset_global_points=True)

        # --------------------- For fusing points ----------------------------------
        if self.rasterizer.global_points is None:
            # Run surfel construction network to build the init surfel.
            self.subforward_recon_surfel_module(input_data, img_wh, reset_global_points=True)
        else:
            if incre_data is None:
                incre_data = input_data
            skf = incre_data.get('skip_flag', False)
            if skf:
                print("Skip!!!")
                return
            self.subforward_fusion_points_module(incre_data, img_wh)
        return None

    def _eval_crop_edge(self, input_dict, gt, crop_size=0, w=640, h=480):
        if crop_size == 0:
            return input_dict, gt
        else:
            pred = input_dict['coarse_color']
            mask = input_dict['ray_mask']

    def _crop_input_coarse_image(self, rendered_output, gt, crop_size=0):
        if crop_size == 0:
            return rendered_output, gt
        else:
            B, H, W, _ = gt.size()
            coarse_color = rendered_output['coarse_color'][:, crop_size: H - crop_size, crop_size: W - crop_size, ]
            output = {'coarse_color': coarse_color}
            gt = gt[:, crop_size: H - crop_size, crop_size: W - crop_size, ]
            return output, gt

    # For evaling rendering
    def render_from_global_model(self, input, **kwargs):
        input_data, incre_data, _ = self._setup_input(input)
        # img_wh = self.cfg.DATA.IMG_WH
        target_camera = self._build_camera_object(input_data['camera'])
        sampled_ray_dir = input_data['sampled_ray']
        sampled_pixel = input_data['sampled_pidx']
        h, w, _ = sampled_ray_dir.size()
        sampled_ray_dir = sampled_ray_dir.reshape(1, h * w, 1, 3)
        rendered_output, sampled_output = self._render_one_image(target_camera, sampled_pixel, sampled_ray_dir,
                                                                 rgb_gt=input_data['gt_rgb_img'])
        if 'no_loss' in kwargs:
            log_vars = dict()
            psrn_valid_pixel = -100
            pass
        else:
            # pdb.set_trace()
            gt_reshaped = input_data['gt_rgb_img'].reshape(rendered_output['coarse_color'].size())
            losses_dict = self.loss_func.compute_color_l2loss(rendered_output, gt_reshaped,)
            psrn_valid_pixel = mse2psnr(losses_dict['loss_color'])
            # print(losses_dict)
            # print("PSNR: {}".format(psrn_valid_pixel))

            rendered_output['coarse_color'] = rendered_output['coarse_color'].reshape(input_data['gt_rgb_img'].size())
            # -------------------- Computing loss ----------------------------------
            # pdb.set_trace()
            rendered_output1, gt = self._crop_input_coarse_image(rendered_output, input_data['gt_rgb_img'], crop_size=6)
            losses_dict = self.loss_func.compute_color_l2loss(rendered_output1, gt,
                                                              need_mask=False, )

            losses_total, log_vars = self._parse_losses(losses_dict)

        output_final = dict()
        output_final['log_vars'] = log_vars
        output_final['gt_image'] = input_data['gt_rgb_img']
        output_final['sampled_output'] = sampled_output
        output_final['rendered_output'] = rendered_output
        output_final['psnr_valid_pixel'] = psrn_valid_pixel
        return output_final

    def forward_render(self, input, **kwargs):
        first_data, second_data, third_data = self._setup_input(input)
        img_wh = self.cfg.DATA.IMG_WH
        #
        surfel_input = {
            'input_image': first_data['rgb'],
            'image_wh': img_wh,
            'pcd_dirs': first_data['data']['pcd_dirs']
        }
        global_points_feats, _ = self.point_recon_net.obtain_points_from_img(surfel_input)
        self.setup_global_points(first_data['data'], features=global_points_feats, reset_global_points=True)
        #

        if second_data is None:
            second_data = first_data
        if third_data is None:
            third_data = second_data
        target_camera = self._build_camera_object(third_data['camera'])
        sampled_ray_dir = third_data['sampled_ray']
        sampled_pixel = third_data['sampled_pidx']
        h, w, _ = sampled_ray_dir.size()
        sampled_ray_dir = sampled_ray_dir.reshape(1, h * w, 1, 3)
        rendered_output, sampled_output = self._render_one_image(target_camera, sampled_pixel, sampled_ray_dir,
                                                                 rgb_gt=third_data['gt_rgb_img'])
        if 'no_loss' in kwargs:
            pass
        else:
            # -------------------- Computing loss ----------------------------------
            losses_dict = self.loss_func.compute_color_l2loss(rendered_output, third_data['gt'])
            losses_total, log_vars = self._parse_losses(losses_dict)
            if 'no_grad' in kwargs:
                pass
            else:
                # optimize
                self._setup_zero_grad()
                if losses_total.data.item() == 0:
                    # pdb.set_trace()
                    print("There is no valid ray!!!")
                else:
                    try:
                        losses_total.backward()
                    except:
                        pdb.set_trace()
                self._step_grad()
        output_final = dict()
        # output_final['log_vars'] = log_vars
        # output_final['gt_image'] = second_data['rgb']
        output_final['sampled_output'] = sampled_output
        output_final['rendered_output'] = rendered_output
        # output_final['gt_pixel'] = second_data['gt']
        return output_final

    def _setup_input(self, input, **kwargs):
        # pdb.set_trace()
        num_input = 0
        if 0 in input:
            num_input += 1
        if 1 in input:
            num_input += 2
        if 2 in input:
            num_input += 5
        #
        if num_input == 1:
            data_item = input[0]
            self._move_dict_to_device(data_item)

            # to device
            return data_item, None, None
        elif num_input == 3:
            data_item1 = input[0]
            data_item2 = input[1]
            self._move_dict_to_device(data_item1)
            self._move_dict_to_device(data_item2)
            return data_item1, data_item2, None
        elif num_input == 8:
            data_item1 = input[0]
            data_item2 = input[1]
            data_item3 = input[2]
            if len(data_item1) < 1:
                data_item1 = None
            else:
                self._move_dict_to_device(data_item1)
            # self._move_dict_to_device(data_item1)
            # pdb.set_trace()
            if len(data_item2) < 1:
                data_item2 = None
            else:
                self._move_dict_to_device(data_item2)
            if len(data_item3) < 1:
                data_item3 = None
            else:
                self._move_dict_to_device(data_item3)
            # self._move_dict_to_device(data_item3)
            return data_item1, data_item2, data_item3
        elif len(input) > 3 and 0 in input and 1 in input and 2 in input:
            data_item1 = input[0]
            data_item2 = input[1]
            data_item3 = input[2]
            if len(data_item1) < 1:
                data_item1 = None
            else:
                self._move_dict_to_device(data_item1)
            if len(data_item2) < 1:
                data_item2 = None
            else:
                self._move_dict_to_device(data_item2)
            if len(data_item3) < 1:
                data_item3 = None
            else:
                self._move_dict_to_device(data_item3)
            # self._move_dict_to_device(data_item2)
            # self._move_dict_to_device(data_item3)
            return data_item1, data_item2, data_item3
        else:
            # pdb.set_trace()
            raise RuntimeError("Wrong.")

    def _build_camera_object(self, cam_para):
        # check input dim
        R = cam_para['R']
        R = R.unsqueeze(0)
        if len(R.size()) == 4:
            R = R.squeeze(0)
        T = cam_para['T']
        T = T.unsqueeze(0)
        if len(T.size()) == 3:
            T = T.squeeze(0)
        camera = FoVPerspectiveCameras(R=R, T=T, fov=cam_para['fov'],
                                       znear=self.cfg.DATA.NEAR_PLANE,
                                       zfar=self.cfg.DATA.FAR_PLANE,
                                       aspect_ratio=cam_para.get('aspect_ratio', 1))
        if torch.isnan(camera.T[0,0]):
            pdb.set_trace()
        return camera.to(self.device)

    def _move_dict_to_device(self, input_dict):
        # output = dict()
        for key in input_dict.keys():
            cont = input_dict[key]
            if torch.is_tensor(cont):
                input_dict[key] = cont.float().to(self.device)
            elif isinstance(cont, dict):
                self._move_dict_to_device(cont)

    def forward(self, input, func_name="forward_train", **kwargs):
        if func_name == "forward_train":
            output = self.forward_train(input, **kwargs)
            self.update_learning_rate()
            return output
        elif func_name == 'forward_fuse':
            output = self.forward_fuse(input, **kwargs)
            return output
        elif func_name == "render_from_global_model":
            output = self.render_from_global_model(input, **kwargs)
            return output
        elif func_name == "forward_per_scene_opt":
            output = self.forward_per_scene_opt(input, **kwargs)
            return output

    def setup_optimizer(self):
        # Optimizers
        self.optimizers = dict()
        if self.cfg.TRAIN.TRAIN_POINTS:
            trainable_paras = []
            trainable_names = []
            for name, para in self.rasterizer.named_parameters():
                if para.requires_grad:
                    trainable_paras.append(para)
                    trainable_names.append(name)
            print("Trainable parameters of points: ", trainable_names)
            # pdb.set_trace()
            self.optimizers['rasterizer'] = torch.optim.Adam(self.rasterizer.parameters(), self.cfg.TRAIN.LEARNING_RATE_POINTS,
                                                             betas=(0.9, 0.999))

        if self.cfg.MODEL.FUSION.BUILD_FUSION:
            self.optimizers['fusion'] = torch.optim.Adam(self.fusion_net.parameters(),
                                                         self.cfg.TRAIN.LEARNING_RATE_FUSION,
                                                         betas=(0.9, 0.999))
        if self.cfg.MODEL.DEPTH.BUILD_DEPTH:
            self.optimizers['depth'] = torch.optim.Adam(self.depth_net.parameters(),
                                                         self.cfg.TRAIN.LEARNING_RATE_DEPTH,
                                                         betas=(0.9, 0.999))


        # if 'Simple' in self.neural_points.__str__() or self.neural_points is None:
        #     print("No learnable parameters in nerual points modules!")
        # else:
        #     self.optimizers['net'] = torch.optim.Adam(self.neural_points.parameters(), self.cfg.TRAIN.LEARNING_RATE,
        #                                               betas=(0.9, 0.999))
        self.optimizers['points'] = torch.optim.Adam(self.aggregator.parameters(), self.cfg.TRAIN.LEARNING_RATE,
                                                     betas=(0.9, 0.999))
        if self.cfg.TRAIN.OPTIMIZER_IMG_ENCODER:
            self.optimizers['point_recon'] = torch.optim.Adam(self.point_recon_net.parameters(),
                                                              self.cfg.TRAIN.LEARNING_RATE,
                                                              betas=(0.9, 0.999))

    # ====
    # Packages for rendering
    # ====
    def _get_RT_lookat(self, lookat, cam_pos, up, **kwargs):
        skip_flag = False
        R, T = look_at_view_transform(eye=cam_pos, at=lookat, up=up)
        if torch.isnan(T[0, 0]):
            print('look_at: ', lookat)
            print('campos:', cam_pos)
            print('up:', up)
            skip_flag = True
        # R is (world to camera).T -- > so R is the c2w hhhhh
        return R.squeeze(0), T.squeeze(0), skip_flag
