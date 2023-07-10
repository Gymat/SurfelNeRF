import copy
import os
import math
import pdb

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
from .surfelnerf import PointsSurfelModel
from models.depth_reg.depth_regressor import Depth_Regressor_Simple_Model
from models.depth_reg.depth_to_point import Depth2Point
from typing import List
from collections import OrderedDict, namedtuple

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__

@MODEL_REGISTRY.register()
class PointsSurfelModel_wGRU_v1(PointsSurfelModel):
    """
    This class has a GRU to handle the update of surfel
    """
    def __init__(self, cfg, **kwargs):
        super(PointsSurfelModel_wGRU_v1, self).__init__(cfg, **kwargs)
        assert cfg.MODEL.FUSION.BUILD_FUSION, "This class must have a Fusion Network..."

    def fuse_to_global_feats(self, cand_local_features, cand_local_weights,
                             cand_global_weights, cand_global_features,
                             filter_g_features, merge_global_index, filter_global_index, cand_local_dirs,
                             cand_global_dirs):
        # For GRU
        input_weights_emb = positional_encoding(cand_local_weights, self.cfg.MODEL.FUSION.WEIGHTS_DIM)
        global_weights_emb = positional_encoding(cand_global_weights, self.cfg.MODEL.FUSION.WEIGHTS_DIM)
        fused_features = self.fusion_net.forward(input_feat=cand_local_features,
                                                 hidden_feat=cand_global_features,
                                                 input_weights_emb=input_weights_emb,
                                                 hidden_weights_emb=global_weights_emb,
                                                 input_dirs=cand_local_dirs,
                                                 hidden_dirs=cand_global_dirs,
                                                 stage='fuse to global; update points'
                                                 )
        #
        filter_g_features[merge_global_index] = fused_features
        new_global_feats = torch.zeros_like(self.rasterizer.global_features)
        # wo_filter_index = self.__find_wo_filtered_index(filter_global_index, self.rasterizer.global_features.size(1))
        new_global_feats[0, ~filter_global_index] = self.rasterizer.global_features[0, ~filter_global_index]
        new_global_feats[0, filter_global_index] = filter_g_features
        #
        self.rasterizer.global_features = new_global_feats

    def setup_global_points_gru_feats(self, features, weights):
        global_weights_emb = positional_encoding(weights, self.cfg.MODEL.FUSION.WEIGHTS_DIM)
        features_new = self.fusion_net(input_feat=features,
                                       hidden_feat=None,
                                       input_weights_emb=global_weights_emb,
                                       hidden_weights_emb=None,
                                       input_dirs=None,
                                       hidden_dirs=None,
                                       stage='setup globals')
        return features_new


    def forward_train(self, input, **kwargs):
        # get key params

        need_point_con = input.pop('need_point_recon', True)
        # pdb.set_trace()
        if need_point_con and self.cfg.MODEL.FUSION.USE_GRU_DDP:
            if self.scene_name is None:
                self.scene_name = input[0]['data']['scene']
            else:
                scene_now = input[0]['data']['scene']
                new_scene_flag = input.pop('new_scene_flag', False)
                if scene_now == self.scene_name:
                    if new_scene_flag:
                        need_point_con = True
                    else:
                        need_point_con = False
                        input[1] = input[0]  # give original first data to second data
                else:
                    self.scene_name = scene_now

        # need_fuse = input.pop('need_fuse', True)
        if self.rasterizer.global_points is not None and self.cfg.MODEL.FUSION.USE_GRU_DDP:
            self.rasterizer.detach_points()
        #
        # pdb.set_trace()
        return super(PointsSurfelModel_wGRU_v1, self).forward_train(input, need_point_con=need_point_con,
                                                                    **kwargs)

    def fuse_with_new_points_gru_feats(self, local_feats, weights):
        # This func for GRU func, since incremental features need to be fed into the GRU module.
        # This is same as the process in the self.setup_global_points for new surfels.
        return self.setup_global_points_gru_feats(local_feats, weights)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if 'neural_points' in name:
                    continue
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))
        else:
            print('Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            print('Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

@MODEL_REGISTRY.register()
class PointsSurfelModel_wGRU_v1_FT(PointsSurfelModel):
    """
    This class has a GRU to handle the update of surfel
    """
    def __init__(self, cfg, **kwargs):
        super(PointsSurfelModel_wGRU_v1_FT, self).__init__(cfg, build=False, **kwargs)
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
        else:
            raise RuntimeError("This class must have a Fusion Network...")

        ### init optimizer
        if self.cfg.IS_TRAIN:
            self.setup_optimizer()
            self.setup_scheduler()
        # since dataload is multiple process, so we need to put the scene name in here!
        self.scene_name = None

    def setup_optimizer(self):
        # Optimizers
        self.optimizers = dict()
        # pdb.set_trace()
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

        # if self.cfg.MODEL.FUSION.BUILD_FUSION:
        #     self.optimizers['fusion'] = torch.optim.Adam(self.fusion_net.parameters(),
        #                                                  self.cfg.TRAIN.LEARNING_RATE_FUSION,
        #                                                  betas=(0.9, 0.999))
        # if self.cfg.MODEL.DEPTH.BUILD_DEPTH:
        #     self.optimizers['depth'] = torch.optim.Adam(self.depth_net.parameters(),
        #                                                  self.cfg.TRAIN.LEARNING_RATE_DEPTH,
        #                                                  betas=(0.9, 0.999))


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
