import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..helpers.networks import init_seq, positional_encoding
from ..helpers.geometrics import compute_world2local_dist
from models.aggregators.build import AGGREGATORS_REGISTRY
from utils.simple_func import strlist_to_floatlist

@AGGREGATORS_REGISTRY.register()
class PointAggregator_Surfel(torch.nn.Module):

    def __init__(self, cfg):
        super(PointAggregator_Surfel, self).__init__()
        self.act = getattr(nn, cfg.MODEL.AGGREGATOR.ACT_TYPE, None)
        print("opt.act_type!!!!!!!!!", cfg.MODEL.AGGREGATOR.ACT_TYPE)
        self.point_hyper_dim = cfg.MODEL.NEURAL_POINTS.CHANNELS

        self.cfg = cfg
        self.dist_dim = cfg.MODEL.AGGREGATOR.DIST_DIM
        # default dist_func is 'linear', so this is self.linear
        assert self.linear is not None, "InterpAggregator doesn't have disance_kernel {} ".format(cfg.agg_distance_kernel)

        self.agg_axis_weight = strlist_to_floatlist(cfg.MODEL.AGGREGATOR.AGG_AXIS_WEIGHT)
        self.axis_weight = None if self.agg_axis_weight is None else torch.as_tensor(self.agg_axis_weight, dtype=torch.float32, device="cuda")[None, None, None, None, :]

        self.num_freqs = cfg.MODEL.AGGREGATOR.NUM_POS_FREQS if cfg.MODEL.AGGREGATOR.NUM_POS_FREQS > 0 else 0
        self.num_viewdir_freqs = cfg.MODEL.AGGREGATOR.NUM_VIEWDIR_FREQS if cfg.MODEL.AGGREGATOR.NUM_VIEWDIR_FREQS > 0 else 0

        self.pnt_channels = (2 * self.num_freqs * 3) if self.num_freqs > 0 else 3
        self.viewdir_channels = (2 * self.num_viewdir_freqs * 3) if self.num_viewdir_freqs > 0 else 3
        self.shading_color_channel_num = 3
        self.apply_pnt_mask = cfg.MODEL.AGGREGATOR.APPLY_PNT_MASK
        self.dist_xyz_freq = cfg.MODEL.AGGREGATOR.DIST_XYZ_FREQ
        self.sparse_loss_weight = cfg.MODEL.AGGREGATOR.SPARSE_LOSS_WEIGHT
        self.zero_one_loss_items = cfg.LOSS.ZERO_LOSS_ITEMS

        self.viewmlp_init()

        self.density_super_act = torch.nn.Softplus()
        self.density_act = torch.nn.ReLU()
        self.color_act = torch.nn.Sigmoid()

        ## magic number
        self.act_super = True

    def raw2out_density(self, raw_density):
        if self.act_super:
            # return self.density_act(raw_density - 1)  # according to mip nerf, to stablelize the training
            return self.density_super_act(raw_density - 1)  # according to mip nerf, to stablelize the training
        else:
            return self.density_act(raw_density)

    def raw2out_color_f(self, raw_color):
        color = self.color_act(raw_color)  # sigmoid
        if self.act_super:
            color = color * (1 + 2 * 0.001) - 0.001  # according to mip nerf, to stablelize the training
        return color

    def viewmlp_init(self):

        block_init_lst = []
        dist_xyz_dim = 1 * abs(5) * self.dist_dim  # still magic number ??? # dist_dim is the dimension of distance in
        # the world/camera coordinates
        """ Setup the parameters """
        ### for the input channels
        point_features_dim = self.cfg.MODEL.NEURAL_POINTS.CHANNELS  # TODO: double check the params here
        num_feat_freqs = self.cfg.MODEL.AGGREGATOR.NUM_FEAT_FREQS
        agg_intrp_order = self.cfg.MODEL.AGGREGATOR.AGG_INTRP_ORDER
        shading_feature_mlp_layer1 = self.cfg.MODEL.AGGREGATOR.SHADING_FEATURE_MLP_LAYER1  # 2
        shading_feature_mlp_layer3 = self.cfg.MODEL.AGGREGATOR.SHADING_FEATURE_MLP_LAYER3  # 2
        point_color_mode = self.cfg.MODEL.AGGREGATOR.POINT_COLOR_MODE
        point_dir_mode = self.cfg.MODEL.AGGREGATOR.POINT_DIR_MODE
        out_channels = self.cfg.MODEL.AGGREGATOR.SHADING_OUT_CHANNEL
        shading_alpha_mlp_layer = self.cfg.MODEL.AGGREGATOR.SHADING_ALPHA_MLP_LAYER
        shading_color_mlp_layer = self.cfg.MODEL.AGGREGATOR.SHADING_COLOR_MLP_LAYER
        self.surfel_pos_enc_num = self.cfg.MODEL.AGGREGATOR.SURFEL_POS_ENC_FREQ


        in_channels = point_features_dim  # 32
        # in_channels = (in_channels) 32 + (2 * num_feat_freqs * in_channels) 192 [This is the position embedding] +
        # dist_xyz_dim 60 [containing the distance in world coordinates and camera coordiantes, 30 + 30]
        in_channels += (2 * num_feat_freqs * in_channels) + 6


        if shading_feature_mlp_layer1 > 0:
            block1 = []
            for i in range(shading_feature_mlp_layer1):
                block1.append(nn.Linear(in_channels, out_channels))
                block1.append(self.act(inplace=True))
                in_channels = out_channels
            self.block1 = nn.Sequential(*block1)
            block_init_lst.append(self.block1)
        else:
            self.block1 = self.passfunc


        if shading_feature_mlp_layer3 > 0:
            in_channels = in_channels + (3 if "1" in list(point_color_mode) else 0) + (
                9 if "1" in list(point_dir_mode) else 0)
            in_channels += 2*2*self.surfel_pos_enc_num
            in_channels += 1
            block3 = []
            for i in range(shading_feature_mlp_layer3):
                block3.append(nn.Linear(in_channels, out_channels))
                block3.append(self.act(inplace=True))
                in_channels = out_channels
            self.block3 = nn.Sequential(*block3)
            block_init_lst.append(self.block3)
        else:
            self.block3 = self.passfunc

        # final alpha block
        alpha_block = []
        in_channels_alpha = out_channels
        out_channels_alpha = int(out_channels / 2)
        for i in range(shading_alpha_mlp_layer - 1):
            alpha_block.append(nn.Linear(in_channels_alpha, out_channels_alpha))
            alpha_block.append(self.act(inplace=False))
            in_channels_alpha = out_channels_alpha
        alpha_block.append(nn.Linear(in_channels_alpha, 1))
        self.alpha_branch = nn.Sequential(*alpha_block)
        block_init_lst.append(self.alpha_branch)

        color_block = []
        in_channels_color = out_channels + self.viewdir_channels
        out_channels_color = int(out_channels)
        for i in range(shading_color_mlp_layer - 1):
            color_block.append(nn.Linear(in_channels_color, out_channels_color))
            color_block.append(self.act(inplace=True))
            in_channels_color = out_channels_color
        color_block.append(nn.Linear(in_channels_color, 3))
        self.color_branch = nn.Sequential(*color_block)
        block_init_lst.append(self.color_branch)

        for m in block_init_lst:
            init_seq(m)

    def passfunc(self, input):
        return input

    def avg(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * channel* R * SR * K
        # return B * R * SR * K
        weights = pnt_mask * 1.0
        return weights, embedding

    def linear(self, embedding, dists, pnt_mask, vsize, grid_vox_sz, axis_weight=None):
        # dists: B * R * SR * K * channel
        # return B * R * SR * K
        if axis_weight is None or (axis_weight[..., 0] == 1 and axis_weight[..., 2] ==1):
            weights = 1. / torch.clamp(torch.norm(dists[..., :3], dim=-1), min= 1e-6)
        else:
            weights = 1. / torch.clamp(torch.sqrt(torch.sum(torch.square(dists[...,:2]), dim=-1)) * axis_weight[..., 0] + torch.abs(dists[...,2]) * axis_weight[..., 1], min= 1e-6)
        weights = pnt_mask * weights
        return weights, embedding

    def gradiant_clamp(self, sampled_conf, min=0.0001, max=1):
        diff = sampled_conf - torch.clamp(sampled_conf, min=min, max=max)
        return sampled_conf - diff.detach()

    def forward(self, sampled_embedding, sampled_color, dists_to_ray, sampled_pcd_dir, sampled_ray_dir,
                valid_ray_pnt_mask, sampled_pos_x, sampled_pos_y):
        '''

        :param sampled_embedding: (B, R, SR, K, channels), where R is the number of ray, K is the number of sampled points per rays.
        :param dists_to_ray: (B, R, SR, K, 1)
        :param sampled_pcd_dir: (B, R, SR, K, 3)
        :param sampled_ray_dir: (B, R, 1, 3)
        :param valid_ray_pnt_mask: (B, Hs, Ws, SR, K), where value is True/False, Hs and Ws indicate sampled height and width
        :param sampled_pos_x/y: (B, R, SR, K, 1)
        :return:
        '''
        num_feat_freqs = self.cfg.MODEL.AGGREGATOR.NUM_FEAT_FREQS  # 3
        # obtain size of tensor
        B, R, SR, K, _ = sampled_embedding.size()
        valid_pnt_mask = valid_ray_pnt_mask.reshape(-1)
        valid_local_block_mask = torch.any(valid_ray_pnt_mask, dim=-1).reshape(B*R*SR)
        #
        sampled_ray_dir_expanded = sampled_ray_dir.expand(B, R, SR*K, 3)
        sampled_ray_dir_expanded = sampled_ray_dir_expanded.reshape(B*R*SR, K, 3)
        # sampled_ray_dir_expanded_feat1 = sampled_ray_dir.expand(B, R, SR*K, 3)
        # change size of input tensor to (-1, channels)
        sampled_embedding = sampled_embedding.reshape(-1, sampled_embedding.size(-1))  # (B*R*SR*K, feats channels)
        sampled_color = sampled_color.reshape(-1, sampled_color.size(-1))
        dists_to_ray = dists_to_ray.reshape(-1, dists_to_ray.size(-1))
        sampled_pcd_dir = sampled_pcd_dir.reshape(-1, sampled_pcd_dir.size(-1))
        sampled_ray_dir_expanded_1 = sampled_ray_dir_expanded.reshape(-1, sampled_ray_dir.size(-1))

        # obtain pos emb
        pos_feat = torch.cat([sampled_pos_x, sampled_pos_y], dim=-1)
        pos_feat = pos_feat.reshape(-1, 2)  # (B*R*SR*K, 2)
        # pdb.set_trace()
        pos_enc_surfel = positional_encoding(pos_feat, self.surfel_pos_enc_num)

        # obtain weights
        weights = 1./ torch.clamp(dists_to_ray, min=1e-6)
        weights = weights.reshape(B*R*SR, K)
        weights = weights / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1e-8)
        weights = weights.unsqueeze(-1)

        # obtain valid sampled feats
        valid_embedding = sampled_embedding[valid_pnt_mask, :]
        valid_color = sampled_color[valid_pnt_mask, :]
        valid_dis2ray = dists_to_ray[valid_pnt_mask, :]
        valid_pcd_dir = sampled_pcd_dir[valid_pnt_mask, :]
        valid_surfel_pos_enc = pos_enc_surfel[valid_pnt_mask, :]
        valid_ray_dir_expanded = sampled_ray_dir_expanded[valid_local_block_mask, 0, :]
        valid_ray_dir_pnt_expanded = sampled_ray_dir_expanded_1[valid_pnt_mask, :]


        # pdb.set_trace()
        ray_dir_posemb = positional_encoding(valid_ray_dir_expanded, self.num_viewdir_freqs)
        valid_ray_posemb = ray_dir_posemb.reshape(-1, ray_dir_posemb.size(-1))
        '''block 1'''
        dist_feat = positional_encoding(valid_dis2ray, num_feat_freqs)
        feat = torch.cat([valid_embedding, positional_encoding(valid_embedding, num_feat_freqs), dist_feat], dim=-1)
        # pdb.set_trace()
        feat1 = self.block1(feat)
        '''block 3'''
        # pdb.set_trace()
        angle_two_views = torch.sum(valid_pcd_dir * valid_ray_dir_pnt_expanded, dim=-1, keepdim=True)
        feat3_input = torch.cat([feat1, valid_color, valid_pcd_dir, valid_ray_dir_pnt_expanded,
                                 valid_pcd_dir - valid_ray_dir_pnt_expanded, valid_surfel_pos_enc, angle_two_views], dim=-1)
        feat3 = self.block3(feat3_input)  # feat3: torch.Size([1 * 1024 * 8, 256]), where is BS x R x K x channel
        '''Render'''
        # for density
        density = self.raw2out_density(self.alpha_branch(feat3))  # torch.Size([1 * 1024 * 8, 1])
        density_holder = torch.zeros(B * R * SR * K, density.size(-1), dtype=torch.float32, device=feat3.device)
        density_holder[valid_pnt_mask, :] = density
        density = density_holder.view(B*R*SR, K, density_holder.shape[-1])  # (B*R*SR, K, 1)
        # pdb.set_trace()
        density_weighted = density * weights  # torch.Size([1 * 1024 * 8, 1])
        density = density_weighted.sum(dim=-2)
        valid_density = density.view(-1, density.shape[-1])[valid_local_block_mask]
        # for color
        feat3_holder = torch.zeros([B * R * SR * K, feat3.shape[-1]], dtype=torch.float32, device=feat3.device)
        feat3_holder[valid_pnt_mask, :] = feat3
        feat3_new = feat3_holder.view(B * R * SR, K, feat3_holder.shape[-1])
        feat_color = (feat3_new * weights).sum(dim=-2)
        feat_color = feat_color.view([-1, feat3_new.shape[-1]])[valid_local_block_mask]
        # feat_color = torch.sum(feat3_new * weights, dim=-2).view([-1, feat3_new.shape[-1]])[ray_valid, :]
        color_in1 = torch.cat([feat_color, valid_ray_posemb], dim=-1)
        # pdb.set_trace()
        color_out1 = self.color_branch(color_in1)
        color_output = self.raw2out_color_f(color_out1)
        output = torch.cat([valid_density, color_output], dim=-1)
        #
        output_placeholder = torch.zeros([B*R*SR, self.shading_color_channel_num + 1], dtype=torch.float32,
                                         device=output.device)
        output_placeholder[valid_local_block_mask] = output
        return output_placeholder, weights

    def _linear_weight_from_dist(self, dists_to_ray, valid_ray_mask):
        weights = 1. / torch.clamp(dists_to_ray, min=1e-6)
        weights = weights * valid_ray_mask.unsqueeze(2)
        return weights

