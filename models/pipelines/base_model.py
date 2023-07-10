import torch
from torch import nn
import os
from collections import OrderedDict, namedtuple
import torch.distributed as dist
from typing import List

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__

class BaseModel(nn.Module):
    def __init__(self, cfg, is_train=True, **kwargs):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.is_train = is_train
        #

        self.save_dir = os.path.join(cfg.LOG.CHECKPOINT_PATH, cfg.LOG.TASK_NAME)
        torch.backends.cudnn.benchmark = True

        self.loss_names = []  # losses to report
        self.model_names = []  # models that will be used
        self.visual_names = []  # visuals to show at test time


    def name(self):
        return self.__class__.__name__

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
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    # def initialize(self, opt):
    #     self.opt = opt
    #     self.gpu_ids = opt.gpu_ids
    #     self.is_train = opt.is_train
    #     self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]) if self.
    #                                gpu_ids else torch.device('cpu'))
    #     self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    #     torch.backends.cudnn.benchmark = True
    #
    #     self.loss_names = []  # losses to report
    #     self.model_names = []  # models that will be used
    #     self.visual_names = []  # visuals to show at test time

    def set_input(self, input: dict):
        self.input = input

    def setup(self, opt):
        '''Creates schedulers if train, Load and print networks if resume'''
        if not self.is_train or opt.resume_dir:
            print("opt.resume_iter!!!!!!!!!", opt.resume_iter)
            self.load_networks(opt.resume_iter)
        self.print_networks(opt.verbose)

    def eval(self):
        '''turn on eval mode'''
        for net in self.get_networks():
            net.eval()

    def train(self):
        for net in self.get_networks():
            net.train()

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_networks(self) -> [nn.Module]:
        ret = []
        for name in self.model_names:
            assert isinstance(name, str)
            net = getattr(self, 'net_{}'.format(name))
            assert isinstance(net, nn.Module)
            ret.append(net)
        return ret

    def get_current_visuals(self, data=None):
        ret = {}
        for name in self.visual_names:
            assert isinstance(name, str)
            if name not in ["gt_image_ray_masked", "ray_depth_masked_gt_image", "ray_depth_masked_coarse_raycolor", "ray_masked_coarse_raycolor"]:
                ret[name] = getattr(self, name)
        if "coarse_raycolor" not in self.visual_names:
            ret["coarse_raycolor"] = getattr(self, "coarse_raycolor")
        return ret

    def get_current_losses(self):
        ret = {}
        for name in self.loss_names:
            assert isinstance(name, str)
            ret[name] = getattr(self, 'loss_' + name)
        return ret

    def save_networks(self, epoch, other_states={}, back_gpu=True):
        for name, net in zip(self.model_names, self.get_networks()):
            save_filename = '{}_net_{}.pth'.format(epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)

            try:
                if isinstance(net, nn.DataParallel):
                    net = net.module
                net.cpu()
                torch.save(net.state_dict(), save_path)
                if back_gpu:
                    net.cuda()
            except Exception as e:
                print("savenet:", e)

        save_filename = '{}_states.pth'.format(epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(other_states, save_path)

    def load_networks(self, epoch):
        for name, net in zip(self.model_names, self.get_networks()):
            print('loading', name)
            assert isinstance(name, str)
            load_filename = '{}_net_{}.pth'.format(epoch, name)
            load_path = os.path.join(self.opt.resume_dir, load_filename)

            if not os.path.isfile(load_path):
                print('cannot load', load_path)
                continue

            state_dict = torch.load(load_path, map_location=self.device)
            if isinstance(net, nn.DataParallel):
                net = net.module

            net.load_state_dict(state_dict, strict=False)


    def print_networks(self, verbose):
        print('------------------- Networks -------------------')
        for name, net in zip(self.model_names, self.get_networks()):
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network {}] Total number of parameters: {:.3f}M'.format(
                name, num_params / 1e6))
        print('------------------------------------------------')

    def set_requires_grad(self, nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self, **kwargs):
        for key in self.schedulers.keys():
            self.schedulers[key].step()
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
                # print('optimizer {}, learning rate = {:.7f}'.format(i + 1, lr))

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

