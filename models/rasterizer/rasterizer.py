import pdb
import pickle
import argparse
import os
import sys
import torch
import torch.nn as nn
from DSS_EXT.DSS.surfel_core.rasterizer_base import SurfaceSplatting, PointsRasterizationSettings
from DSS_EXT.DSS.surfel_core.rasterizer_base_hw_v2 import PointsRasterizationSettings_hw
from DSS_EXT.DSS.surfel_core.rasterizer_base_hw_v2 import SurfaceSplatting_hw_v2
from DSS_EXT.DSS.surfel_core.cloud import PointClouds3D
from .build import RASTERIZER_REGISTRY

@RASTERIZER_REGISTRY.register()
class Rasterize_Surfel():
    def __init__(self, cfg, **kwargs):
        super(Rasterize_Surfel, self).__init__()
        self.cfg = cfg
        self.device = cfg.DEVICE
        self._build_init()

    def _build_init(self):
        self.global_points = None  # Batch(1) x N x 3
        self.global_normals = None
        self.global_features = None
        self.global_weights = None
        self.global_colors = None
        self.global_dirs = None
        self.depth_merge_thres = self.cfg.depth_merge_thres
        if self.cfg.MODEL.RASTERIZER.BIN_SIZE == 8888:
            # check
            assert self.cfg.MODEL.RASTERIZER.POINTS_PER_PIXEL == self.cfg.MODEL.RASTERIZER.points_per_local * self.cfg.MODEL.RASTERIZER.num_shading_points
        self.rasterizer_setting = PointsRasterizationSettings(image_size=self.cfg.img_size,
                                                              points_per_pixel=self.cfg.MODEL.RASTERIZER.POINTS_PER_PIXEL,
                                                              bin_size=self.cfg.MODEL.RASTERIZER.BIN_SIZE,
                                                              points_per_local=self.cfg.MODEL.RASTERIZER.points_per_local,
                                                              num_shaping_points=self.cfg.MODEL.RASTERIZER.num_shading_points
                                                              )
        # self.cam_center = self.cam_params['cam_center']
        # self.cam_focal = self.cam_params['focal']
        pass

    def setup_global_model(self, points, normals, features, colors=None,  pcd_dirs=None, weights=None, points_mask=None,
                           reset=False):
        if points_mask is not None:
            if points_mask.dtype != torch.bool:
                points_mask = points_mask.bool()
            # pdb.set_trace()
            assert points_mask.dtype == torch.bool
            # check size() of points_mask
            if len(points_mask.size()) == 3 and points_mask.size(0) == 1:
                points_mask = points_mask.squeeze(0)
            if len(points_mask.size()) == 2 and points_mask.size(1) == 1:
                points_mask = points_mask.squeeze(1)
            else:
                raise RuntimeError("The size of points_mask is wrong.")
            points = points[:, points_mask]
            colors = colors[:, points_mask]
            pcd_dirs = pcd_dirs[:, points_mask]
            normals = normals[:, points_mask]
            features = features[:, points_mask]
            weights = weights[:, points_mask]
        if reset:
            self.global_points = points.to(self.device)
            self.global_normals = normals.to(self.device)
            self.global_features = features.to(self.device)
            self.global_weights = weights.to(self.device)
            self.global_colors = colors.to(self.device)
            self.global_dirs = pcd_dirs.to(self.device)
            return
        if self.global_points is None:
            self.global_points = points.to(self.device)
        else:
            raise RuntimeError("There are existing points in the model.")
        if self.global_normals is None:
            self.global_normals = normals.to(self.device)
        if self.global_features is None:
            self.global_features = features.to(self.device)
        if self.global_weights is None:
            self.global_weights = weights.to(self.device)
        if self.global_colors is None:
            self.global_colors = colors.to(self.device)
        if self.global_dirs is None:
            self.global_dirs = pcd_dirs.to(self.device)

    def _build_rasterizer(self, camera):
        Rasterizer = SurfaceSplatting(cameras=camera, raster_settings=self.rasterizer_setting)
        return Rasterizer

    def rasterize(self, target_camera):
        # pdb.set_trace()
        # This is for call to obtain the output fragments
        global_model = PointClouds3D(points=self.global_points, normals=self.global_normals,
                                     features=self.global_features, colors=self.global_colors,
                                     dirs=self.global_dirs, weights=self.global_weights)
        Rasterizer = self._build_rasterizer(target_camera)
        Rasterizer.to(self.device)
        output = Rasterizer.forward(global_model)
        return output

    def update_global_points(self, add_points, add_normals, add_feats, add_weights, add_colors=None, add_dirs=None,):
        self.global_points = torch.cat([self.global_points, add_points], dim=1)
        self.global_normals = torch.cat([self.global_normals, add_normals], dim=1)
        try:
            self.global_features = torch.cat([self.global_features, add_feats], dim=1)
        except:
            self.global_features = torch.cat([self.global_features, add_feats.unsqueeze(0)], dim=1)
        self.global_weights = torch.cat([self.global_weights, add_weights], dim=1)
        self.global_colors = torch.cat([self.global_colors, add_colors], dim=1)
        self.global_dirs = torch.cat([self.global_dirs, add_dirs], dim=1)

    def return_point_cloud(self):
        pcd = {
            'points': self.global_points.cpu(), 'normals': self.global_normals.cpu(), 'features': self.global_features.cpu(),
            'weights': self.global_weights.cpu(), 'colors': self.global_colors.cpu(), 'dirs': self.global_dirs.cpu()
        }
        return pcd


@RASTERIZER_REGISTRY.register()
class Rasterize_Surfel_HW(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Rasterize_Surfel_HW, self).__init__()
        self.cfg = cfg
        self.device = cfg.DEVICE
        self._build_init()

    def _load_init_points(self):
        path = os.path.join(self.cfg.MODEL.RASTERIZER.load_init_point_root,
                            self.cfg.MODEL.RASTERIZER.load_init_point_name.format(self.cfg.MODEL.RASTERIZER.load_init_point_iter, self.cfg.DATA.SCENE_NAME))
        if self.cfg.old_load_points:
            data = torch.load(path)
            print("Load points from ", path)
        else:
            print("The points be should put in {}".format(path))
            data = torch.load(self.cfg.MODEL.RASTERIZER.load_init_point_path,)
            # raise UserWarning("The load point way is updated !!! Please pay attention to here!!! ")
            print("Load points from ", self.cfg.MODEL.RASTERIZER.load_init_point_path)
        points_dict = data['points']
        self.setup_learnable_global_model(**points_dict)

    def _build_init(self):
        self.depth_merge_thres = self.cfg.depth_merge_thres
        if self.cfg.MODEL.RASTERIZER.BIN_SIZE == 8888:
            # check
            assert self.cfg.MODEL.RASTERIZER.POINTS_PER_PIXEL == self.cfg.MODEL.RASTERIZER.points_per_local * self.cfg.MODEL.RASTERIZER.num_shading_points
        self.rasterizer_setting = PointsRasterizationSettings_hw(image_size_h=self.cfg.img_size_h,
                                                                 image_size_w=self.cfg.img_size_w,
                                                              points_per_pixel=self.cfg.MODEL.RASTERIZER.POINTS_PER_PIXEL,
                                                              bin_size=self.cfg.MODEL.RASTERIZER.BIN_SIZE,
                                                              points_per_local=self.cfg.MODEL.RASTERIZER.points_per_local,
                                                              num_shaping_points=self.cfg.MODEL.RASTERIZER.num_shading_points
                                                              )
        # self.cam_center = self.cam_params['cam_center']
        # self.cam_focal = self.cam_params['focal']
        if self.cfg.MODEL.RASTERIZER.load_init_point:
            print("Load init points!")
            self._load_init_points()
        else:
            self.global_points = None  # Batch(1) x N x 3
            self.global_normals = None
            self.global_features = None
            self.global_weights = None
            self.global_colors = None
            self.global_dirs = None

    def setup_global_model(self, points, normals, features, colors=None,  pcd_dirs=None, weights=None, points_mask=None,
                           reset=False):
        if points_mask is not None:
            if points_mask.dtype != torch.bool:
                points_mask = points_mask.bool()
            # pdb.set_trace()
            assert points_mask.dtype == torch.bool
            # check size() of points_mask
            if len(points_mask.size()) == 3 and points_mask.size(0) == 1:
                points_mask = points_mask.squeeze(0)
            if len(points_mask.size()) == 2 and points_mask.size(1) == 1:
                points_mask = points_mask.squeeze(1)
            else:
                raise RuntimeError("The size of points_mask is wrong.")
            points = points[:, points_mask]
            colors = colors[:, points_mask]
            pcd_dirs = pcd_dirs[:, points_mask]
            normals = normals[:, points_mask]
            features = features[:, points_mask]
            weights = weights[:, points_mask]
        if reset:
            self.global_points = points.to(self.device)
            self.global_normals = normals.to(self.device)
            self.global_features = features.to(self.device)
            self.global_weights = weights.to(self.device)
            self.global_colors = colors.to(self.device)
            self.global_dirs = pcd_dirs.to(self.device)
            return
        if self.global_points is None:
            self.global_points = points.to(self.device)
        else:
            raise RuntimeError("There are existing points in the model.")
        if self.global_normals is None:
            self.global_normals = normals.to(self.device)
        if self.global_features is None:
            self.global_features = features.to(self.device)
        if self.global_weights is None:
            self.global_weights = weights.to(self.device)
        if self.global_colors is None:
            self.global_colors = colors.to(self.device)
        if self.global_dirs is None:
            self.global_dirs = pcd_dirs.to(self.device)

    def _build_rasterizer(self, camera):
        Rasterizer = SurfaceSplatting_hw_v2(cameras=camera, raster_settings=self.rasterizer_setting)
        return Rasterizer

    def rasterize(self, target_camera):
        # pdb.set_trace()
        # This is for call to obtain the output fragments
        global_model = PointClouds3D(points=self.global_points, normals=self.global_normals,
                                     features=self.global_features, colors=self.global_colors,
                                     dirs=self.global_dirs, weights=self.global_weights)
        Rasterizer = self._build_rasterizer(target_camera)
        Rasterizer.to(self.device)
        output = Rasterizer.forward(global_model)
        return output

    def update_global_points(self, add_points, add_normals, add_feats, add_weights, add_colors=None, add_dirs=None,):
        self.global_points = torch.cat([self.global_points, add_points], dim=1)
        self.global_normals = torch.cat([self.global_normals, add_normals], dim=1)
        try:
            self.global_features = torch.cat([self.global_features, add_feats], dim=1)
        except:
            self.global_features = torch.cat([self.global_features, add_feats.unsqueeze(0)], dim=1)
        self.global_weights = torch.cat([self.global_weights, add_weights], dim=1)
        self.global_colors = torch.cat([self.global_colors, add_colors], dim=1)
        self.global_dirs = torch.cat([self.global_dirs, add_dirs], dim=1)

    def return_point_cloud(self):
        pcd = {
            'points': self.global_points.cpu(), 'normals': self.global_normals.cpu(), 'features': self.global_features.cpu(),
            'weights': self.global_weights.cpu(), 'colors': self.global_colors.cpu(), 'dirs': self.global_dirs.cpu()
        }
        return pcd

    def setup_learnable_global_model(self, points, normals, features, weights, colors, dirs,
                                     requires_grad=True):
        # self.register_parameter('global_points', nn.Parameter(points, requires_grad=requires_grad,))
        # self.register_parameter('global_normals', nn.Parameter(normals, requires_grad=requires_grad,))
        # self.register_parameter('global_features', nn.Parameter(features, requires_grad=requires_grad, ))
        # self.register_parameter('global_colors', nn.Parameter(colors, requires_grad=requires_grad, ))
        # self.register_parameter('global_dirs', nn.Parameter(dirs, requires_grad=requires_grad, ))
        self.global_points = nn.Parameter(points, requires_grad=False,)
        self.global_normals = nn.Parameter(normals, requires_grad=False)
        self.global_features = nn.Parameter(features, requires_grad=requires_grad)
        self.global_weights = nn.Parameter(weights, requires_grad=False)
        self.global_colors = nn.Parameter(colors, requires_grad=requires_grad)
        self.global_dirs = nn.Parameter(dirs, requires_grad=requires_grad)

    def detach_points(self):
        self.global_points = self.global_points.detach()
        self.global_normals = self.global_normals.detach()
        self.global_features = self.global_features.detach()
        self.global_weights = self.global_weights.detach()
        self.global_colors = self.global_colors.detach()
        self.global_dirs = self.global_dirs.detach()
