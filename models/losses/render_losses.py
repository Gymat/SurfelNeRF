import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import format as fmt
from utils.simple_func import strlist_to_floatlist, str_to_strlist

class BaseRenderLosses(nn.Module):
    def __init__(self, cfg):
        super(BaseRenderLosses, self).__init__()
        self.cfg = cfg
        self.opt = cfg.LOSS
        self.device = cfg.DEVICE
        self.is_train = cfg.IS_TRAIN

        # loss weights
        # color
        self.color_loss_items = str_to_strlist(self.opt.COLOR_LOSS_ITEMS)
        self.color_loss_weights = strlist_to_floatlist(self.opt.COLOR_LOSS_WEIGHT)
        # depth
        self.depth_loss_items = str_to_strlist(self.opt.DEPTH_LOSS_ITEMS)
        self.depth_loss_weights = strlist_to_floatlist(self.opt.DEPTH_LOSS_WEIGHT)
        # sparse
        self.zero_one_loss_items = str_to_strlist(self.opt.ZERO_ONE_LOSS_ITEMS)
        self.zero_one_loss_weights = strlist_to_floatlist(self.opt.ZERO_ONE_LOSS_WEIGHT)

        # add the functions used in losses
        self.l1loss = torch.nn.L1Loss().to(self.device)
        self.l2loss = torch.nn.MSELoss().to(self.device)

    def compute_color_l2loss(self, output, gt, need_mask=True, crop_size=0):
        '''

        :param output: {"coarse_color": (B, R, 3), "ray_mask": (B, R)}
        :param gt: (B, R, 3)
        :return:
        '''
        loss_dict = dict()
        #
        pred_color = output['coarse_color']
        if need_mask:
            masked = output['ray_mask']
            masked_output = pred_color[masked, ...]
            masked_gt = gt[masked, ...]
        else:
            masked_output = pred_color
            masked_gt = gt
        loss = self.l2loss(masked_output, masked_gt)
        #
        loss_dict['loss_color'] = loss
        return loss_dict

    def compute_losses(self, output, input_data):
        loss_dict = dict()
        # for the color loss

        loss_color = self._compute_color_loss(output, input_data['gt_image'])
        loss_dict['loss_color'] = loss_color
        # for sparse zero one loss
        loss_sparse_zero_one, flag = self._compute_sparse_zero_one_loss(output)
        if flag:
            loss_dict['loss_sparse'] = loss_sparse_zero_one

        return loss_dict

    def _compute_color_loss(self, output, gt_color_image):
        loss_color = 0
        flag = False
        # pdb.set_trace()
        for i, name in enumerate(self.color_loss_items):
            if name.startswith("ray_masked"):
                unmasked_name = name[len("ray_masked")+1:]
                has_occ_ray = (output["ray_mask"] > 0)
                masked = has_occ_ray[..., None].expand(-1, -1, 3)
                # masked_rato = has_occ_ray.sum().item() / output["ray_mask"].size(-1)
                # print("valid ray rato: {}/{}".format(has_occ_ray.sum().item(), output["ray_mask"].size(-1)))
                # check the masked rato !
                if self.is_train:
                    masked_output = torch.masked_select(output[unmasked_name], masked).reshape(1, -1, 3)
                    masked_gt = torch.masked_select(gt_color_image, masked).reshape(1, -1, 3)
                else:
                    masked_output = output[unmasked_name].reshape(1, -1, 3)
                    masked_gt = gt_color_image.reshape(1, -1, 3)
                    # pdb.set_trace()
                if masked_output.shape[1] > 0:
                    loss = self.l2loss(masked_output.to(masked_gt.device), masked_gt)
                    # loss_regular = 0.1 * self._color_different_loss(masked_output)
                    # loss = loss + loss_regular
                else:
                    loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)
            else:
                if name not in output:
                    print(fmt.YELLOW + "No required color loss item: " + name +
                          fmt.END)
                    # raise RuntimeError()
                # print("no_mask")
                loss = self.l2loss(output[name], self.gt_image)

            # total loss
            loss_color = loss_color + loss * self.color_loss_weights[i]

        return loss_color

    def _color_different_loss(self, masked_output,):
        loss = 1 - self.l1loss(masked_output[..., 0], masked_output[..., 1]) - self.l1loss(masked_output[..., 0], masked_output[..., 2]) - self.l1loss(masked_output[..., 1], masked_output[..., 2])
        return loss


    def _compute_depth_loss(self, output, gt_depth):
        raise NotImplemented()

    def _compute_sparse_zero_one_loss(self, output, ):
        loss_total = 0
        flag = False
        if len(self.zero_one_loss_items) == 0:
            return 0, flag
        # zero_one regularization losses
        for i, name in enumerate(self.zero_one_loss_items):
            if name not in output:
                pass
                # pdb.set_trace()
                # print(fmt.YELLOW + "No required zero_one loss item: " + name +
                #       fmt.END)
                # setattr(self, "loss_" + name, torch.zeros([1], device="cuda", dtype=torch.float32))
            else:
                val = torch.clamp(output[name], 1e-3,
                                  1 - 1e-3)
                # print("self.output[name]",torch.min(self.output[name]), torch.max(self.output[name]))
                loss = torch.mean(torch.log(val) + torch.log(1 - val))
                loss_total = loss_total + loss * self.zero_one_loss_weights[i]
                flag = True

        return loss_total, flag






