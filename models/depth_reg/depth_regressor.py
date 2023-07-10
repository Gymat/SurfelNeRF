import pdb

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.distributed as dist
from .architectures import Unet, Unet_64, Unet_128
# from models.pipelines.base_model import BaseModel
# from models.helpers.networks import get_scheduler
from collections import OrderedDict
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
# from models.networks.cost_volum_unet import Unet_Cost_Volume
# from models.networks.coding import PositionalEncoding

# def get_depth_regressor(opt):
#     if opt.depth_regressor == "unet":
#         return  Depth_Regressor(opt)
#

class Depth_Regressor_Simple_Model(nn.Module):
    # THis model provides forward func only!
    def __init__(self, cfg):
        super(Depth_Regressor_Simple_Model, self).__init__()
        self.cfg = cfg
        self.pts_regressor = Depth_Regressor(cfg)
        self.min_z = cfg.DATA.NEAR_PLANE
        self.gt_depth_loss_weight = 1.0
        #
        if cfg.MODEL.DEPTH.WEIGHTS is not None:
            model = torch.load(cfg.MODEL.DEPTH.WEIGHTS)
            # pdb.set_trace()
            self.load_state_dict_cus(model['weights'], strict=False)

    def load_self_weights_only(self):
        if self.cfg.MODEL.DEPTH.WEIGHTS is not None:
            model = torch.load(self.cfg.MODEL.DEPTH.WEIGHTS)
            # pdb.set_trace()
            self.load_state_dict_cus(model['weights'], strict=False)


    def forward(self, depth_img, input_img, need_loss=True, **kwargs):
        '''

        :param depth_img:  (B, C, H, W)
        :param input_img:  (B, C, H, W)
        :param kwargs:
        :return:
        '''
        # pdb.set_trace()
        inverse_depth_img = 1. / torch.clamp(depth_img, min=0.001)
        inverse_depth_img[depth_img < 0.001] = 0
        ref_depth = inverse_depth_img.detach()

        inverse_depth_img = inverse_depth_img / (1.0 / self.min_z)
        #
        # pdb.set_trace()
        depth_input = torch.cat((input_img, inverse_depth_img), dim=1)
        #
        regressed_pts, refine_depth = self.pts_regressor(depth_input,)
        # eval loss
        if need_loss:
            depth_loss_value = self.depth_loss(depth_img, regressed_pts)
            # d = {'depth_loss': depth_loss_value}
        else:
            depth_loss_value = 0

        return regressed_pts, refine_depth, depth_loss_value

    def depth_loss(self, gt_depth, regressed_pts):

        with torch.no_grad():
            valid_depth_mask = gt_depth > 0.0
        #
        # pdb.set_trace()
        depth_loss = nn.L1Loss()(regressed_pts[valid_depth_mask],
                                  gt_depth[valid_depth_mask]) * self.gt_depth_loss_weight
        return depth_loss

    def load_state_dict_cus(self, state_dict: 'OrderedDict[str, Tensor]',
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
        return None





class Depth_Regressor_Model(nn.Module):
    def __init__(self, cfg):
        super(Depth_Regressor_Model, self).__init__()
        self.cfg = cfg
        self.pts_regressor = Depth_Regressor(cfg)
        self.min_z = cfg.DATA.NEAR_PLANE
        #
        self.gt_depth_loss_weight = 3.0
        self.setup_optimizer()

    def _setup_zero_grad(self):
        for key in self.optimizers.keys():
            self.optimizers[key].zero_grad()

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # Allow setting None for some loss item.
            # This is to support dynamic loss module, where the loss is
            # calculated with a fixed frequency.
            elif loss_value is None:
                continue
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        # Note that you have to add 'loss' in name of the items that will be
        # included in back propagation.
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            # pdb.set_trace()
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def _step_grad(self):
        for key in self.optimizers.keys():
            self.optimizers[key].step()

    def forward(self, depth_img, input_img, **kwargs):
        '''

        :param depth_img:  (B, C, H, W)
        :param input_img:  (B, C, H, W)
        :param kwargs:
        :return:
        '''
        inverse_depth_img = 1. / torch.clamp(depth_img, min=0.001)
        inverse_depth_img[depth_img < 0.001] = 0
        ref_depth = inverse_depth_img.detach()

        inverse_depth_img = inverse_depth_img / (1.0 / self.min_z)
        #

        depth_input = torch.cat((input_img, inverse_depth_img), dim=1)
        #
        regressed_pts, refine_depth = self.pts_regressor(depth_input,)
        # eval loss
        depth_loss = self.depth_loss(depth_img, regressed_pts)
        #
        losses_dict = {'depth_loss': depth_loss}
        losses_total, log_vars = self._parse_losses(losses_dict)
        #
        self._setup_zero_grad()
        if self.cfg.IS_TRAIN:
            losses_total.backward()
            #
            self._step_grad()
        #
        output = {
            'regressed_pts': regressed_pts,
            'log_vars': log_vars,
            'refine_depth': refine_depth,
            'ref_depth': ref_depth
        }

        return output

    def setup_optimizer(self):
        self.optimizers = dict()
        self.optimizers['depth'] =torch.optim.Adam(self.pts_regressor.parameters(), self.cfg.TRAIN.LEARNING_RATE,
                                                      betas=(0., 0.999))
        pass

    def depth_loss(self, gt_depth, regressed_pts):

        with torch.no_grad():
            valid_depth_mask = gt_depth > 0.0
        #
        depth_loss = nn.L1Loss()(regressed_pts[valid_depth_mask],
                                  gt_depth[valid_depth_mask]) * self.gt_depth_loss_weight
        return depth_loss

    def update_learning_rate(self, **kwargs):
        # for key in self.schedulers.keys():
        #     self.schedulers[key].step()
        for i, optim_name in enumerate(self.optimizers.keys()):
            optim = self.optimizers[optim_name]
            lr = optim.param_groups[0]['lr']
            if "opt" in kwargs:
                opt = kwargs["opt"]
                if not opt.lr_policy.startswith("iter") or \
                        ("total_steps" in kwargs and kwargs["total_steps"] % opt.print_freq == 0):
                    # print('optimizer {}, learning rate = {:.7f}'.format(i + 1, lr))
                    pass
            else:
                pass


class Depth_Regressor(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.min_z = cfg.DATA.NEAR_PLANE
        # if opt.depth_com:
        channels_in = 4
        # else:
        #     channels_in = 3
        # if opt.est_opacity:
        #     channels_out = 2
        # else:
        channels_out = 1

        self.model = Unet(channels_in=channels_in, channels_out=channels_out, opt=cfg)
        self.pad = 256
        #
        # if opt.regressor_model=="Unet":
        #
        # elif opt.regressor_model == "Unet64":
        #     self.model = Unet_64(channels_in=channels_in, channels_out=channels_out, opt=opt)
        #     self.pad = 64
        # elif opt.regressor_model == "Unet128":
        #     self.model = Unet_128(channels_in=channels_in, channels_out=channels_out, opt=opt)
        #     self.pad = 128
        # self.opt = opt

    def forward(self, depth_input, input_RTs=None,  K=None):

        B, C, H, W = depth_input.shape
        if H % self.pad == 0 and W % self.pad == 0:
            regressed_depth = self.model(depth_input)
        else:
            padding_H = math.ceil( H / self.pad) * self.pad
            padding_W = math.ceil( W / self.pad) * self.pad
            padding = max(padding_H, padding_W)
            depth_input = F.pad(depth_input, (0, padding - W, 0, padding - H), mode='constant', value=0)

            regressed_depth = self.model(depth_input)

            regressed_depth = regressed_depth[:, :, 0:H, 0:W]

        output = nn.Sigmoid()(regressed_depth)
        regressed_pts = output[:, 0:1]

        # opacity = None

        if True:
            if True:
                regressed_pts = regressed_pts * ( 1.0 / self.min_z)
            refine_depth = regressed_pts.detach()
            regressed_pts = 1. / torch.clamp(regressed_pts, min=0.001)
        # else:
        #     regressed_pts = (
        #         regressed_pts
        #         * (self.opt.max_z - self.opt.min_z)
        #         + self.opt.min_z
        #     )
        #     refine_depth = regressed_pts.detach()
        return regressed_pts, refine_depth