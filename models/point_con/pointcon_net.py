import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.build import build_backbone
from models.backbone.PointMlp import premlp_init
from models.point_con.voxelize import construct_vox_points_closest, obtain_voxel_size_based_grid_res, incre_points
from utils.simple_func import strlist_to_floatlist


class PointRecon(nn.Module):
    '''
    Point Recon main class.
    '''

    def __init__(self, cfg):
        super(PointRecon, self).__init__()
        self.cfg = cfg
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
        self.voxel_size = torch.FloatTensor(strlist_to_floatlist(self.cfg.MODEL.NEURAL_POINTS.VOX_SIZE))
        #
        self.min_bbox = None


    def project_point_feats(self, input):
        '''
        This func takes image features and depth map as input,
        and outputs the points position with corresponding features.
        Since the depth map is in the camera coordinates, so we also need a cam2world matrix
        :param input:
        :return:
        '''
        # setup input
        img_feats = input['img_feats']  # multi-scale image features [(H, W, 3)[RGB color], (H, W, D)[x1],
        #                                                             (H/2, W/2, D), (H/4, W/4, D)]
        img_num = img_feats[0].size(0)
        # pcd_points = input['pcd_points']  # pcd position (H*W, 3)
        imgh, imgw = input['size']['h'], input['size']['w']
        #
        # concat
        point_img_feats = []

        for index, feat_maps in enumerate(img_feats):
            if index == 0:
                color_rgb = feat_maps.detach()
            # point_img_feats.append(feat_maps)
            B, _, fmaph, fmapw = feat_maps.size()
            assert (imgh // fmaph == imgw // fmapw) and (imgh % fmaph == 0), "Size of feature map is wrong!"
            scale_factor = imgh // fmaph
            if torch.is_tensor(scale_factor):
                scale_factor = scale_factor.item()
            if scale_factor > 1:
                up_sample_feat_maps = F.interpolate(feat_maps, scale_factor=scale_factor, mode="bilinear",
                                                    align_corners=True)
            else:
                up_sample_feat_maps = feat_maps
            point_img_feats.append(up_sample_feat_maps)
        #
        point_img_feats = torch.cat(point_img_feats, dim=1)  # B x C x H x W
        point_img_feats = point_img_feats.permute(0, 2, 3, 1)
        try:
            point_img_feats = point_img_feats.reshape((1, imgh*imgw*img_num, self.img_feat_channels))  # (B, H*W, dim)
        except:
            torch.cuda.empty_cache()
            point_img_feats = point_img_feats.cpu()
            point_img_feats = point_img_feats.reshape(
                (1, imgh * imgw * img_num, self.img_feat_channels))  # (B, H*W, dim)
            point_img_feats = point_img_feats.to(feat_maps.device)
            pdb.set_trace()
        color_rgb = color_rgb.permute(0, 2, 3, 1)
        color_rgb = color_rgb.reshape((1, imgh*imgw*img_num, 3))

        return point_img_feats, color_rgb

    def forward(self, input, **kwargs):
        '''

        :param input:
        :param kwargs:
        :return:
        '''
        # First, we extract the 2d image features
        input_img = input['input_image']  # size: (1, Number of images, 3, height, width)
        img_size_wh = input['image_wh']
        points_pos = input['points_pos']  # torch.Tensor, B = 1
        points_mask = input['points_mask']  # where points_mask=1 indicates the valid position!
        device = points_pos.device
        input_img.to(device)

        # Then, we voxelize the point cloud
        valid_index = torch.where(points_mask == 1)[:2]
        # valid_points_pos = points_pos[valid_index]
        # valid_points_pos = valid_points_pos.unsqueeze(0)

        points_pos = torch.index_select(points_pos, dim=1, index=valid_index[1])

        if self.cfg.MODEL.NEURAL_POINTS.VOXEL_RES > 0:
            voxel_size_, min_xyz = obtain_voxel_size_based_grid_res(points_pos, voxel_res=self.cfg.MODEL.NEURAL_POINTS.VOXEL_RES)
            # print("Voxel size:", voxel_size_)
            self.voxel_size = voxel_size_

        sparse_id_list, min_idx_list = construct_vox_points_closest(points_pos,
                                                                    voxel_size=self.voxel_size.to(device),
                                                                    ranges=self.ranges.to(device)
                                                                    )
        min_idx_list = min_idx_list[0]
        # Note: if key frame is too much or larger, we need to split them and then concat them during evaluation
        if 'max_batch' in kwargs:
            raise NotImplementedError("coming soon.")
            pdb.set_trace()
            # total_batch = input_img.size(1)
            # max_batch = kwargs['max_batch']
            # batch = total_batch // max_batch
            # points_feats_list, color_rgb_list = [], []
            # for ibatch in range(batch):
            #     if max_batch * (ibatch + 1) > total_batch:
            #         end = total_batch
            #     else:
            #         end = max_batch * (ibatch + 1)
            #     input_subimg = input_img[:, max_batch*ibatch:end, ]
            #     img_feats_list = self.img_enc(input_subimg)  # B x N_imgs x H x W
            #     back_project_input = {
            #         "img_feats": img_feats_list, "size": {'w': img_size_wh[0], 'h': img_size_wh[1]}
            #     }
            #     points_feats, color_rgb = self.project_point_feats(back_project_input)  # (H*w, Dim=)
            #     points_feats_list.append(points_feats.cpu())
            #     color_rgb_list.append(color_rgb.cpu())
            # points_feats = torch.cat(points_feats_list, dim=0)
            # color_rgb = torch.cat(color_rgb_list, dim=0)
            # pdb.set_trace()
            # points_feats = points_feats.to(input_img.device)
            # color_rgb = color_rgb.to(input_img.device)


        else:
            # min_idx_list : torch.Tensor, (1, id)
            # Obtain points features from image features
            img_feats_list = self.img_enc(input_img)  # 1 x B x N_imgs x H x W
            back_project_input = {
                "img_feats": img_feats_list, "size": {'w': img_size_wh[0], 'h': img_size_wh[1]}
            }
            try:
                points_feats, color_rgb = self.project_point_feats(back_project_input)  # (H*w, Dim=)
                points_feats = torch.index_select(points_feats, dim=1, index=valid_index[1])
                # points_feats = points_feats[valid_index]
                color_rgb = torch.index_select(color_rgb, dim=1, index=valid_index[1])
            except:
                pdb.set_trace()

            #
            points_dir = input['points_dir']  # (B, h*w, 3)
            points_dir.to(device)
            points_dir = torch.index_select(points_dir, dim=1, index=valid_index[1])
            # points_dir = torch.index_select(points_dir, )
            #
            points_input_emb = torch.cat([points_feats, points_dir], dim=-1)

            selected_points_emb = torch.index_select(points_input_emb, dim=1, index=min_idx_list)
            selected_points_pos = torch.index_select(points_pos, dim=1, index=min_idx_list)
            selected_color_rgb = torch.index_select(color_rgb, dim=1, index=min_idx_list)
            selected_points_dir = torch.index_select(points_dir, dim=1, index=min_idx_list)
            # selected_points_emb = torch.cat([selected_points_feats, selected_points_dir], dim=-1)
        #
        points_embedding = self.point_mlp.forward(selected_points_emb)  # B x N_{points} x C

        return points_embedding.squeeze(0), selected_points_pos.squeeze(0), selected_color_rgb.squeeze(
            0), selected_points_dir.squeeze(0)

    def forward_LotsofImages(self, input, **kwargs):
        '''

        :param input:
        :param kwargs:
        :return:
        '''
        # First, we extract the 2d image features
        input_img = input['input_image']  # size: (1, Number of images, 3, height, width)
        img_size_wh = input['image_wh']
        points_pos = input['points_pos']  # torch.Tensor, B = 1
        points_mask = input['points_mask']  # where points_mask=1 indicates the valid position!
        device = points_pos.device
        input_img.to(device)

        # Then, we voxelize the point cloud
        valid_index = torch.where(points_mask == 1)[:2]
        # valid_points_pos = points_pos[valid_index]
        # valid_points_pos = valid_points_pos.unsqueeze(0)

        points_pos = torch.index_select(points_pos, dim=1, index=valid_index[1])

        if self.cfg.MODEL.NEURAL_POINTS.VOXEL_RES > 0:
            voxel_size_, min_xyz = obtain_voxel_size_based_grid_res(points_pos, voxel_res=self.cfg.MODEL.NEURAL_POINTS.VOXEL_RES)
            # print("Voxel size:", voxel_size_)
            self.voxel_size = voxel_size_

        sparse_id_list, min_idx_list = construct_vox_points_closest(points_pos,
                                                                    voxel_size=self.voxel_size.to(device),
                                                                    )
        min_idx_list = min_idx_list[0]
        # Note: if key frame is too much or larger, we need to split them and then concat them during evaluation
        total_batch = input_img.size(1)
        max_batch = 64  # Hard code!!!
        batch = total_batch // max_batch
        if total_batch % max_batch == 0:
            extra = 0
        else:
            extra = 1
        points_feats_list, color_rgb_list = [], []
        for ibatch in range(batch + extra):
            if max_batch * (ibatch + 1) > total_batch:
                end = total_batch
            else:
                end = max_batch * (ibatch + 1)
            input_subimg = input_img[:, max_batch*ibatch:end, ]
            img_feats_list = self.img_enc(input_subimg)  # B x N_imgs x H x W
            back_project_input = {
                "img_feats": img_feats_list, "size": {'w': img_size_wh[0], 'h': img_size_wh[1]}
            }
            points_feats, color_rgb = self.project_point_feats(back_project_input)  # (H*w, Dim=)
            points_feats_list.append(points_feats.cpu())
            color_rgb_list.append(color_rgb.cpu())

        points_feats = torch.cat(points_feats_list, dim=1)
        color_rgb = torch.cat(color_rgb_list, dim=1)
        # pdb.set_trace()
        # points_feats = points_feats.to(input_img.device)
        # color_rgb = color_rgb.to(input_img.device)

        #
        points_dir = input['points_dir']  # (B, h*w, 3)
        # points_dir.to(device)
        points_dir = torch.index_select(points_dir, dim=1, index=valid_index[1])
        color_rgb = torch.index_select(color_rgb, dim=1, index=valid_index[1].cpu())
        points_feats = torch.index_select(points_feats, dim=1, index=valid_index[1].cpu())
        # points_dir = torch.index_select(points_dir, )
        #
        # pdb.set_trace()
        points_dir = points_dir.cpu()
        points_input_emb = torch.cat([points_feats, points_dir], dim=-1)
        min_idx_list = min_idx_list.cpu()
        selected_points_emb = torch.index_select(points_input_emb, dim=1, index=min_idx_list)
        selected_points_pos = torch.index_select(points_pos, dim=1, index=min_idx_list.cuda())
        selected_color_rgb = torch.index_select(color_rgb, dim=1, index=min_idx_list)
        selected_points_dir = torch.index_select(points_dir, dim=1, index=min_idx_list)

        torch.cuda.empty_cache()
        # pdb.set_trace()
        selected_points_emb = selected_points_emb.to(device)
        selected_points_pos = selected_points_pos.to(device)
        selected_color_rgb = selected_color_rgb.to(device)
        selected_points_dir = selected_points_dir.to(device)
        points_embedding = self.point_mlp.forward(selected_points_emb)  # B x N_{points} x C


        return points_embedding.squeeze(0), selected_points_pos.squeeze(0), selected_color_rgb.squeeze(
            0), selected_points_dir.squeeze(0)
