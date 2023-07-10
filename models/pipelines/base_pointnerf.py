import torch
from models.pipelines.base_model import BaseModel
from models.helpers.networks import get_scheduler
from collections import OrderedDict

class PointsBaseModel(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(PointsBaseModel, self).__init__(cfg=cfg, is_train=cfg.IS_TRAIN)
        self.device = torch.device('cuda')
        # bg color
        bg_color = cfg.DATA.BG_COLOR
        if not bg_color or bg_color == 'black':
            bg_color = (0, 0, 0)
        elif bg_color == 'white':
            bg_color = (1, 1, 1)
        self.bg_color = torch.FloatTensor(bg_color, )
        self.bg_color.to(self.device)
        self.neural_points = None

    def _split_ray(self, non_key_frame):
        #
        all_rays = non_key_frame['raydir']  # 1 x 1 x HW x 3
        total_num = all_rays.size()[2]
        ray_num = int(self.cfg.DATA.RANDOM_SAMPLE_SIZE)**2
        split_num = total_num // ray_num
        sampled_ray_list = []
        for i in range(split_num + 1):
            begin_index = ray_num * i
            end_index = ray_num * (i + 1)
            if end_index >= total_num:
                end_index = total_num
            if begin_index >= total_num:
                continue
            sampled_ray_dir = all_rays[:, :, begin_index:end_index]

            sampled_ray_list.append(sampled_ray_dir)
        return sampled_ray_list

    def _step_grad(self):
        for key in self.optimizers.keys():
            self.optimizers[key].step()

    def _setup_input_for_point_recon(self, input):
        key_frame_info = input['key_frame']
        input_for_point_recon = dict()
        input_for_point_recon['points_pos'] = key_frame_info['point_position'].squeeze(1)
        input_for_point_recon['points_dir'] = key_frame_info['point_dir'].squeeze(1)
        input_for_point_recon['image_wh'] = [input['image_wh']['w'], input['image_wh']['h']]
        input_for_point_recon['input_image'] = key_frame_info['rgb_images'].detach()
        input_for_point_recon['points_mask'] = key_frame_info['point_mask'].squeeze(1)  # B x H*W x 1

        return input_for_point_recon

    def _setup_input_for_render(self, input):
        input_for_render = input['train_frame']
        input_for_render['gt_image'] = input_for_render['gt_image'].to(self.device)
        # input_for_render['register_flag'] = input['register_flag']
        input_for_render['register_flag'] = input['register_flag']  # new
        input_for_render['reset_points'] = input['reset_points']
        # check the dim

        return input_for_render

    def setup_optimizer(self):
        # Optimizers
        self.optimizers = dict()
        # pdb.set_trace()
        if 'Simple' in self.neural_points.__str__() or self.neural_points is None:
            print("No learnable parameters in nerual points modules!")
        else:
            self.optimizers['net'] = torch.optim.Adam(self.neural_points.parameters(), self.cfg.TRAIN.LEARNING_RATE,
                                                      betas=(0.9, 0.999))
        self.optimizers['points'] = torch.optim.Adam(self.aggregator.parameters(), self.cfg.TRAIN.LEARNING_RATE_POINTS,
                                                     betas=(0.9, 0.999))
        if self.cfg.TRAIN.OPTIMIZER_IMG_ENCODER:
            self.optimizers['point_recon'] = torch.optim.Adam(self.point_recon_net.parameters(),
                                                              self.cfg.TRAIN.LEARNING_RATE,
                                                              betas=(0.9, 0.999))
            # self.optimizer['point_mlp'] = torch.optim.Adam(self.point_mlp.parameters(), self.cfg.TRAIN.LEARNING_RATE,
            #                                              betas=(0.9, 0.999))

    def setup_scheduler(self):
        self.schedulers = OrderedDict()
        for optim_name in self.optimizers.keys():
            self.schedulers[optim_name] = get_scheduler(self.optimizers[optim_name], self.cfg.TRAIN)

    def _setup_zero_grad(self):
        for key in self.optimizers.keys():
            self.optimizers[key].zero_grad()

    def load_optimizer_state_dict(self, weights_dict):
        #
        for key in self.optimizers.keys():
            if key not in weights_dict:
                print("{} Not in weights dict about optimizer!!!".format(key))
                continue
            self.optimizers[key].load_state_dict(weights_dict[key])
            # change tensor to target device
            for state in self.optimizers[key].state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    def get_optimizer_state_dict(self):
        state_dict_ = dict()
        for key in self.optimizers.keys():
            state_dict_[key] = self.optimizers[key].state_dict()
        return state_dict_

    def get_lr_scheduler_state_dict(self):
        output = dict()
        for key in self.schedulers.keys():
            output[key] = self.schedulers[key].state_dict()
        return output

    def load_scheduler_state_dict(self, weights_dict):
        #
        for key in self.schedulers.keys():
            # pdb.set_trace()
            if key not in weights_dict:
                continue
            self.schedulers[key].load_state_dict(weights_dict[key])
            # change tensor to target device
            # for state in self.schedulers[key].state.values():
            #     for k, v in state.items():
            #         if torch.is_tensor(v):
            #             state[k] = v.cuda()
