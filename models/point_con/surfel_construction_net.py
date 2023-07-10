import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.build import build_backbone
from models.backbone.PointMlp import premlp_init
from utils.simple_func import strlist_to_floatlist


class SurfelRecon(nn.Module):
    '''
    Point Recon main class.
    '''

    def __init__(self, cfg):
        super(SurfelRecon, self).__init__()
        self.cfg = cfg
        self.device = cfg.DEVICE
        # other hparams
        self.ranges = torch.FloatTensor(strlist_to_floatlist(cfg.DATA.RANGES),)
        # channel dict
        channels_dict = {
            'FeatureNet': 59,
            'MnasMulti': 83
        }
        self.img_feat_channels = channels_dict[self.cfg.MODEL.BACKBONE.NAME]

        # init network
        # build the 2d encoder for image
        self.img_enc = build_backbone(cfg)
        # build the init point generator
        self.point_mlp = premlp_init(cfg, self.img_feat_channels + 3)
        #
        # self.voxel_size = torch.FloatTensor(strlist_to_floatlist(self.cfg.MODEL.NEURAL_POINTS.VOX_SIZE))
        #
        self.min_bbox = None

    def forward_w_index(self, input, ):
        input_img = input['input_image']  # size: (1, Number of images, 3, height, width)
        img_size_wh = input['image_wh']
        # points_pos = input['points_pos']  # torch.Tensor, B = 1
        add_local_index = input['add_local_index']
        merge_local_index = input['merge_local_index']
        # points_mask = input['points_mask']  # where points_mask=1 indicates the valid position!
        # device = points_pos.device
        input_img.to(self.device)
        #
        img_feats_list = self.img_enc(input_img)  # 1 x B x N_imgs x H x W

        img_num = img_feats_list[0].size(0)

        back_project_input = {
            "img_feats": img_feats_list, "size": {'w': img_size_wh[0], 'h': img_size_wh[1]},
            'pcd_dirs': input['pcd_dirs']
        }

        add_local_feats, add_local_rgb, merge_local_feats, merge_local_rgb = self.project_point_w_index(
            back_project_input, add_local_index, merge_local_index)
        return add_local_feats, add_local_rgb, merge_local_feats, merge_local_rgb

    def project_point_w_index(self, input, add_local_index=None, merge_local_index=None):
        # setup input
        img_feats = input['img_feats']  # multi-scale image features [(H, W, 3)[RGB color], (H, W, D)[x1],
        #                                                             (H/2, W/2, D), (H/4, W/4, D)]
        pcd_dirs = input['pcd_dirs']
        img_num = img_feats[0].size(0)
        assert img_num == 1
        # pcd_points = input['pcd_points']  # pcd position (H*W, 3)
        imgh, imgw = input['size']['h'], input['size']['w']
        # index
        # add_local_index = input['add_index']
        # merge_local_index = input['merge_local_index']
        # concat
        point_img_feats = []

        for index, feat_maps in enumerate(img_feats):
            if index == 0:
                color_rgb = feat_maps.detach()
            # point_img_feats.append(feat_maps)
            B, _, fmaph, fmapw = feat_maps.size()
            # assert (imgh // fmaph == imgw // fmapw) and (imgh % fmaph == 0), "Size of feature map is wrong!"
            scale_factor = imgh // fmaph
            if torch.is_tensor(scale_factor):
                scale_factor = scale_factor.item()
            if scale_factor > 1:
                up_sample_feat_maps = F.interpolate(feat_maps, size=(imgh, imgw), mode="bilinear",
                                                    align_corners=True)
            else:
                up_sample_feat_maps = feat_maps
            point_img_feats.append(up_sample_feat_maps)
        #
        point_img_feats = torch.cat(point_img_feats, dim=1)  # B x C x H x W
        point_img_feats = point_img_feats.permute(0, 2, 3, 1)  # B x H x W x C
        # point_img_feats = torch.cat([point_img_feats, pcd_dirs], dim=-1)
        point_img_feats = point_img_feats.reshape(point_img_feats.size(0), imgh*imgw, point_img_feats.size(-1))
        point_img_feats = torch.cat([point_img_feats, pcd_dirs], dim=-1)
        #
        color_rgb = color_rgb.permute(0, 2, 3, 1)
        color_rgb = color_rgb.reshape((1, imgh * imgw * img_num, 3))
        if add_local_index is None and merge_local_index is None:
            return point_img_feats, color_rgb, None, None
        # select index
        add_local_feats = point_img_feats[0, add_local_index]
        add_local_feats = self.point_mlp(add_local_feats)
        add_local_rgb = color_rgb[0, add_local_index]
        merge_local_feats = point_img_feats[0, merge_local_index]
        merge_local_rgb = color_rgb[0, merge_local_index]
        merge_local_feats = self.point_mlp(merge_local_feats)

        return add_local_feats, add_local_rgb, merge_local_feats, merge_local_rgb

    def obtain_points_from_img(self, input):
        # First, we extract the 2d image features
        input_img = input['input_image']  # size: (1, Number of images, 3, height, width)
        img_size_wh = input['image_wh']
        pcd_dirs = input['pcd_dirs']  # size: (1, H, W, 3)
        # points_pos = input['points_pos']  # torch.Tensor, B = 1
        # points_mask = input['points_mask']  # where points_mask=1 indicates the valid position!
        input_img.to(self.device)
        #
        img_feats_list = self.img_enc(input_img)  # 1 x B x N_imgs x H x W
        back_project_input = {
            "img_feats": img_feats_list, "size": {'w': img_size_wh[0], 'h': img_size_wh[1]}, 'pcd_dirs': pcd_dirs
        }
        points_feats, color_rgb, _, _ = self.project_point_w_index(back_project_input)  # (H*w, Dim=)
        #
        # points_feats = torch.cat([points_feats, pcd_dirs], dim=-1)
        points_feats_output = self.point_mlp(points_feats)
        return points_feats_output, color_rgb
