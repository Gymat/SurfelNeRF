import torch
import os
import datetime
import time
import glob
import torch.nn as nn
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from utils.simple_func import compute_metrics
from models.pipelines.build import build_model
from data.build import build_dataloader
from utils.visualizer import Visualizer
from ddp_api import comm
from tqdm import tqdm
from lpips import LPIPS
import pdb


class Base_loop_online(nn.Module):
    def __init__(self, config, **kwargs):
        super(Base_loop_online, self).__init__()
        # config
        self.cfg = config
        # common params
        self.epoch = 0
        self.iter_ = 0

        # Time
        TimeObj = datetime.datetime.now()
        strtime = TimeObj.strftime("_%d_%H_%M_%S")
        self.runtime = 'time' + strtime

        # Build the model pipeline
        self.model = build_model(config,
                                 log_path=os.path.join(self.cfg.LOG.CHECKPOINT_PATH, self.cfg.LOG.TASK_NAME))


        self.model.to_gpu()

        # Build the dataloader
        self.data_loader = build_dataloader(cfg=config, **kwargs)

        self.device = torch.device("cuda:0")

        if not os.path.exists(os.path.join(self.cfg.LOG.CHECKPOINT_PATH, self.cfg.LOG.TASK_NAME)):
            os.makedirs(os.path.join(self.cfg.LOG.CHECKPOINT_PATH, self.cfg.LOG.TASK_NAME))
        self.log_path = os.path.join(self.cfg.LOG.CHECKPOINT_PATH, self.cfg.LOG.TASK_NAME)
        if not os.path.exists(os.path.join(self.log_path, 'images')):
            os.mkdir(os.path.join(self.log_path, 'images'))
        # Build the logger
        self.logger = Visualizer(config, self.log_path)
        # dump the config
        cfg_str = config.dump()
        with open(os.path.join(self.log_path, 'ref_pretrained_config.yaml'), 'w') as f:
            f.write(cfg_str)
        f.close()
        # Keep training for using unreliable GPUs
        if self.cfg.KEEP_TRAINING and self.cfg.IS_TRAIN:
            self._load_latest_checkpoints()
        # Load the target checkpoint!
        if config.MODEL.WEIGHTS is not None:
            self.load_checkpoints(config.MODEL.WEIGHTS)

    def train(self, **kwargs):
        iter_ = self.iter_
        epoch = self.epoch
        #
        while(iter_ < self.cfg.TRAIN.MAX_ITER):
            for input_data in self.data_loader:
                skip_flag = input_data.pop('skip_flag', False)
                if skip_flag:
                    print("Skip!!!")
                    torch.cuda.empty_cache()
                    continue
                # pdb.set_trace()
                output_final = self.model.forward(input_data, **kwargs)
                # pdb.set_trace()d
                log_vars = output_final['log_vars']
                # Update the learning rate
                self.model.update_learning_rate()
                # Print the losses
                self.logger.accumulate_losses(log_vars)
                if iter_ % self.cfg.LOG.STEP_PRINT_LOSS == 0:
                    self.logger.print_losses(iter_, )
                    self.logger.reset()
                    torch.cuda.empty_cache()

                # save model only
                if iter_ % self.cfg.LOG.SAVE_ITER_STEP == 0 and iter_ > 0:
                    # save latest checkpoints every epoch!
                    self.save_checkpoints(iter_)

                # update the iterations
                iter_ += 1
                if iter_ >= self.cfg.TRAIN.MAX_ITER:
                    break


                # final iterations
            if iter_ >= self.cfg.TRAIN.MAX_ITER:
                points_checkpoints = self.model.rasterizer.return_point_cloud()
                self.save_checkpoints(iter_, points=points_checkpoints)
                break
            # save latest checkpoints every epoch!
            self.save_checkpoints(iterations=iter_)
            if epoch % self.cfg.LOG.SAVE_POINT_EPOCH == 0 and epoch > 0:
                points_checkpoints = self.model.rasterizer.return_point_cloud()
                self.save_checkpoints(iter_, extra_name='E{}wpoints'.format(epoch), points=points_checkpoints)

            epoch += 1

    def test(self, **kwargs):
        self.model.eval()
        self.logger.reset()
        with torch.no_grad():
            for index, input_data in enumerate(self.data_loader):
                start_time = time.time()
                output_final = self.model.forward_render(input_data, no_loss=True, **kwargs)
                print("Time : {:.2f}s".format((time.time() - start_time)))

                self.save_image(output_final, index=index, epoch=index, time='test_' + self.runtime)




                # self.logger.print_losses(index)
                # self.logger.reset()

    def register_points(self, **kwargs):
        #
        self.model.eval()
        with torch.no_grad():
            iter_ = 0
            for input_data in self.data_loader:
                self.key_frames_info = input_data['keyframe']
                #
                self.key_frames_info['point_position'] = self.key_frames_info['point_position'].to(self.device).detach()
                self.key_frames_info['point_dir'] = self.key_frames_info['point_dir'].to(self.device).detach()
                # self.key_frames_info['image_wh'] = [input['image_wh']['w'], input['image_wh']['h']]
                self.key_frames_info['rgb_images'] = self.key_frames_info['rgb_images'].to(self.device)
                self.key_frames_info['point_mask'] = self.key_frames_info['point_mask'].to(self.device)  # B x H*W x 1
                #
                # pdb.set_trace()
                input_item = {
                    "key_frame": self.key_frames_info, "train_frame": None,
                    "image_wh": input_data['image_wh'],
                    "reset_points": input_data['reset_points'],
                    'register_flag': input_data['register_flag']
                }
                self.model.forward_w_register(input_item)
                '''
                Save the model!
                '''
                # update the iterations
                iter_ += 1
                del input_item

            # save latest checkpoints every epoch!
            points_checkpoints = self.model.neural_points.save_points()
            self._save_latest_checkpoints(epoch=0, iteration=iter_)

    def save_image(self, output, index=0, epoch=0, time='', extra_name=''):
        # gt = output['gt_image'].cpu()
        img = output['rendered_output']['coarse_color']  # tensor
        #
        # gt_img = gt.numpy()
        img = img.cpu().numpy()
        # pdb.set_trace()
        img = img.reshape((self.cfg.DATA.IMG_WH[1], self.cfg.DATA.IMG_WH[0], 3))
        # pdb.set_trace()

        if not os.path.exists(os.path.join(self.log_path, 'images', time,)):
            os.mkdir(os.path.join(self.log_path, 'images', time,))
        save_path = os.path.join(self.log_path, 'images', time,
                                 "index_{}{}_output.png".format(epoch, extra_name))
        self.logger.save_image(img, save_path)
        # if gt_img.shape[0] > 1:
        #     gt_img = gt_img.reshape((self.cfg.DATA.IMG_WH[1], self.cfg.DATA.IMG_WH[0], 3))
        #     save_path = os.path.join(self.log_path, 'images', time,
        #                          "index_{}_gt.png".format(epoch))
        #     self.logger.save_image(gt_img, save_path)
        print("saving Image to {}".format(save_path))
        pass

    def save_checkpoints(self, iterations, extra_name='', **kwargs):
        checkpoints = OrderedDict()
        if isinstance(self.model, (DistributedDataParallel, )):
            model = self.model.module
        else:
            model = self.model
        #
        checkpoints['weights'] = model.state_dict()
        checkpoints['iteration'] = iterations
        #
        # for the optimizer
        checkpoints['optimizer'] = model.get_optimizer_state_dict()
        checkpoints['scheduler'] = model.get_lr_scheduler_state_dict()
        checkpoints['points_info'] = dict()
        # other parameters
        for key in kwargs.keys():
            checkpoints['points_info'][key] = kwargs[key]


        if not os.path.exists(os.path.join(self.log_path)):
            os.mkdir(self.log_path)
        save_path = os.path.join(self.log_path, "{}_{}_checkpoints.pth".format(iterations, extra_name))
        torch.save(checkpoints, save_path)

    def load_checkpoints(self, path):
        weights = torch.load(path)
        model_weights = weights['weights']
        optimizer_weights = weights['optimizer']
        if 'scheduler' in weights and self.cfg.IS_TRAIN:
            self.model.load_scheduler_state_dict(weights['scheduler'])
        #
        self.model.load_state_dict(model_weights, strict=False)
        #
        if self.cfg.IS_TRAIN:
            self.model.load_optimizer_state_dict(optimizer_weights)
        if 'points' in weights and self.cfg.MODEL.NEURAL_POINTS.LOAD_POINTS:
            self.model.neural_points.load_points(weights['points'])
        return weights

    # def _load_latest_checkpoints(self):
    #     latest_path = os.path.join(self.log_path, 'latest_checkpoints.pth')
    #     if os.path.exists(latest_path):
    #         checkpoints = self.load_checkpoints(latest_path)
    #         self.epoch = checkpoints['epoch']
    #         self.iter_ = checkpoints['iteration']
    #     else:
    #         print("There is no latest checkpoints!")

    def _save_latest_checkpoints(self, **kwargs):
        checkpoints = OrderedDict()
        if isinstance(self.model, (DistributedDataParallel,)):
            model = self.model.module
        else:
            model = self.model
        #
        checkpoints['weights'] = model.state_dict()
        #
        # for the optimizer
        checkpoints['optimizer'] = model.get_optimizer_state_dict()
        # other parameters
        for key in kwargs.keys():
            checkpoints[key] = kwargs[key]

        if not os.path.exists(os.path.join(self.log_path)):
            os.mkdir(self.log_path)
        save_path = os.path.join(self.log_path, "latest_checkpoints.pth")
        torch.save(checkpoints, save_path)

    def _load_latest_checkpoints(self, **kwargs):
        file_names = os.listdir(self.log_path)
        max_num = -1
        ckpt_name = None
        for fname in file_names:
            if 'checkpoints' in fname:
                words = fname.split('__')
                if len(words) == 2:
                    iter_ = words[0]
                    try:
                        iter_ = int(iter_)
                    except:
                        iter_ = -100
                    if iter_ > max_num:
                        max_num = iter_
                        ckpt_name = fname
        if ckpt_name is None:
            print("There is no latest checkpoints!!!")
        else:
            print("Loading the latest checkpoint of {}".format(ckpt_name))
            latest_path = os.path.join(self.log_path, ckpt_name)
            if os.path.exists(latest_path):
                checkpoints = self.load_checkpoints(latest_path)
                # self.epoch = checkpoints['epoch']
                self.iter_ = checkpoints['iteration']


class Train_loop_online(nn.Module):
    def __init__(self, config, model, data_loader, **kwargs):
        super(Train_loop_online, self).__init__()
        # config
        self.cfg = config
        # common params
        self.epoch = kwargs.get('epoch', 0)
        self.iter_ = kwargs.get('iteration', 0)

        # Time
        TimeObj = datetime.datetime.now()
        strtime = TimeObj.strftime("_%d_%H_%M_%S")
        self.runtime = 'time' + strtime

        # Build the model pipeline
        self.model = model

        # Build the dataloader
        self.data_loader = data_loader

        self.device = torch.device("cuda")

        if not os.path.exists(os.path.join(self.cfg.LOG.CHECKPOINT_PATH, self.cfg.LOG.TASK_NAME)):
            os.makedirs(os.path.join(self.cfg.LOG.CHECKPOINT_PATH, self.cfg.LOG.TASK_NAME))
        self.log_path = os.path.join(self.cfg.LOG.CHECKPOINT_PATH, self.cfg.LOG.TASK_NAME)
        if not os.path.exists(os.path.join(self.log_path, 'images')) and comm.is_main_process():
            os.mkdir(os.path.join(self.log_path, 'images'))
        # Build the logger
        self.logger = Visualizer(config, self.log_path)

    def train(self, **kwargs):
        iter_ = self.iter_
        epoch = self.epoch
        #
        while(iter_ < self.cfg.TRAIN.MAX_ITER):
            for input_data in self.data_loader:
                # print(iter_)

                skip_flag = input_data.pop('skip_flag', False)
                if skip_flag:
                    print("Skip!!!")
                    torch.cuda.empty_cache()
                    continue
                # pdb.set_trace()
                output_final = self.model.forward(input_data, **kwargs)
                # pdb.set_trace()
                log_vars = output_final['log_vars']
                # Update the learning rate
                # self.model.update_learning_rate()
                # Print the losses
                self.logger.accumulate_losses(log_vars)
                if iter_ % self.cfg.LOG.STEP_PRINT_LOSS == 0 and comm.is_main_process():
                    self.logger.print_losses(iter_, )
                    self.logger.reset()
                    torch.cuda.empty_cache()

                # save model only
                if iter_ % self.cfg.LOG.SAVE_ITER_STEP == 0 and iter_ > 0 and comm.is_main_process():
                    # save latest checkpoints every epoch!
                    self.save_checkpoints(iter_)

                # update the iterations
                iter_ += 1
                if iter_ >= self.cfg.TRAIN.MAX_ITER:
                    break

            # final iterations
            if iter_ >= self.cfg.TRAIN.MAX_ITER and comm.is_main_process():
                # if isinstance(self.model, (DistributedDataParallel,)):
                #     model = self.model.module
                # else:
                #     model = self.model
                # points_checkpoints = model.rasterizer.return_point_cloud()
                self.save_checkpoints(iter_, )
            if iter_ >= self.cfg.TRAIN.MAX_ITER:
                break
            # save latest checkpoints every epoch!
            # self.save_checkpoints(iterations=iter_)
            # if epoch % self.cfg.LOG.SAVE_POINT_EPOCH == 0 and epoch > 0 and comm.is_main_process():
            #     # points_checkpoints = self.model.rasterizer.return_point_cloud()
            #     self.save_checkpoints(iter_,)
            epoch += 1

    def train_per_scene_opt(self, **kwargs):
        iter_ = self.iter_
        epoch = self.epoch
        #
        while(iter_ < self.cfg.TRAIN.MAX_ITER):
            for input_data in self.data_loader:

                skip_flag = input_data.pop('skip_flag', False)
                if skip_flag:
                    print("Skip!!!")
                    torch.cuda.empty_cache()
                    continue

                output_final = self.model.forward(input_data, func_name="forward_per_scene_opt",
                                                  iterations=iter_,**kwargs)
                # pdb.set_trace()
                log_vars = output_final['log_vars']
                # Update the learning rate
                # self.model.update_learning_rate()
                # Print the losses
                self.logger.accumulate_losses(log_vars)
                if iter_ % self.cfg.LOG.STEP_PRINT_LOSS == 0 and comm.is_main_process():
                    self.logger.print_losses(iter_, )
                    self.logger.reset()
                    torch.cuda.empty_cache()

                # save model only
                if iter_ % self.cfg.LOG.SAVE_ITER_STEP == 0 and iter_ > 0 and comm.is_main_process():
                    # save latest checkpoints every epoch!
                    self.save_checkpoints(iter_)
                    self.save_points(iter_, extra_name=self.cfg.MODEL.RASTERIZER.SAVE_SURFEL_NAME)

                # update the iterations
                iter_ += 1
                if iter_ >= self.cfg.TRAIN.MAX_ITER:
                    break

            # final iterations
            if iter_ >= self.cfg.TRAIN.MAX_ITER and comm.is_main_process():
                # if isinstance(self.model, (DistributedDataParallel,)):
                #     model = self.model.module
                # else:
                #     model = self.model
                # points_checkpoints = model.rasterizer.return_point_cloud()
                self.save_checkpoints(iter_, )
                self.save_points(iter_, extra_name=self.cfg.MODEL.RASTERIZER.SAVE_SURFEL_NAME)
            if iter_ >= self.cfg.TRAIN.MAX_ITER:
                break
            # save latest checkpoints every epoch!
            # self.save_checkpoints(iterations=iter_)
            # if epoch % self.cfg.LOG.SAVE_POINT_EPOCH == 0 and epoch > 0 and comm.is_main_process():
            #     # points_checkpoints = self.model.rasterizer.return_point_cloud()
            #     self.save_checkpoints(iter_,)
            epoch += 1

    def eval_per_scene_opt(self, **kwargs):
        self.model.eval()
        iter_ = self.iter_
        epoch = self.epoch
        with torch.no_grad():
            #
            for input_data in self.data_loader:

                skip_flag = input_data.pop('skip_flag', False)
                if skip_flag:
                    print("Skip!!!")
                    torch.cuda.empty_cache()
                    continue

                output_final = self.model.forward(input_data, func_name="forward_per_scene_opt", no_grad=True, **kwargs)
                # pdb.set_trace()
                log_vars = output_final['log_vars']
                # Update the learning rate
                # self.model.update_learning_rate()
                # Print the losses
                self.logger.accumulate_losses(log_vars)
                # if iter_ % self.cfg.LOG.STEP_PRINT_LOSS == 0 and comm.is_main_process():
                self.logger.print_losses(iter_, )
                    # self.logger.reset()
                    # torch.cuda.empty_cache()

                # save model only
                # if iter_ % self.cfg.LOG.SAVE_ITER_STEP == 0 and iter_ > 0 and comm.is_main_process():
                #     # save latest checkpoints every epoch!
                #     self.save_checkpoints(iter_)
                #     self.save_points(iter_, extra_name=self.cfg.MODEL.RASTERIZER.SAVE_SURFEL_NAME)
                # update the iterations
                iter_ += 1

    def build_global_points(self, **kwargs):
        self.model.eval()
        iter_ = 0
        with torch.no_grad():
            for input_data in tqdm(self.data_loader):

                skip_flag = input_data.get('skip_flag', False)
                if skip_flag:
                    print("Skip!!!")
                    # torch.cuda.empty_cache()
                    continue
                #
                train_set_flag = input_data.get('train_set_flag', True)
                if not train_set_flag:
                    # We only load the train set to build global points
                    continue

                output_final = self.model.forward(input_data, func_name='forward_fuse', **kwargs)
                if self.cfg.TEST.SAVE_INTERMEDIATE_POINTS:
                    self.save_points(iter_, extra_name="intermediate_{}".format(self.cfg.MODEL.RASTERIZER.SAVE_SURFEL_NAME))
                # pdb.set_trace()
                # log_vars = output_final['log_vars']
                # Update the learning rate
                # self.model.update_learning_rate()
                # Print the losses
                # self.logger.accumulate_losses(log_vars)
                # if iter_ % self.cfg.LOG.STEP_PRINT_LOSS == 0 and comm.is_main_process():
                #     self.logger.print_losses(iter_, )
                #     self.logger.reset()
                #     torch.cuda.empty_cache()
                iter_ += 1
        # save surfels
        self.save_points(0, extra_name=self.cfg.MODEL.RASTERIZER.SAVE_SURFEL_NAME)

    def build_global_points_log(self, **kwargs):
        self.model.eval()
        iter_ = 0
        with torch.no_grad():
            for input_data in tqdm(self.data_loader):

                skip_flag = input_data.get('skip_flag', False)
                if skip_flag:
                    print("Skip!!!")
                    # torch.cuda.empty_cache()
                    continue
                #
                train_set_flag = input_data.get('train_set_flag', True)
                if not train_set_flag:
                    # We only load the train set to build global points
                    continue

                output_final = self.model.forward(input_data, func_name='forward_fuse_log', **kwargs)
                if self.cfg.TEST.SAVE_INTERMEDIATE_POINTS:
                    self.save_points(iter_, extra_name="intermediate_{}".format(self.cfg.MODEL.RASTERIZER.SAVE_SURFEL_NAME))
                # pdb.set_trace()
                # log_vars = output_final['log_vars']
                # Update the learning rate
                # self.model.update_learning_rate()
                # Print the losses
                # self.logger.accumulate_losses(log_vars)
                # if iter_ % self.cfg.LOG.STEP_PRINT_LOSS == 0 and comm.is_main_process():
                #     self.logger.print_losses(iter_, )
                #     self.logger.reset()
                #     torch.cuda.empty_cache()
                iter_ += 1
        print("Finish!")
        # save surfels
        # self.save_points(0, extra_name=self.cfg.MODEL.RASTERIZER.SAVE_SURFEL_NAME)

    def build_global_points_and_render(self, **kwargs):
        self.model.eval()
        iter_ = 0
        with torch.no_grad():
            for input_data in tqdm(self.data_loader):

                skip_flag = input_data.get('skip_flag', False)
                if skip_flag:
                    print("Skip!!!")
                    # torch.cuda.empty_cache()
                    continue
                #
                train_set_flag = input_data.get('train_set_flag', True)
                if not train_set_flag:
                    # We only load the train set to build global points
                    continue

                output_final = self.model.forward(input_data, func_name='forward_fuse_and_render_one_image', **kwargs)
                self.save_image(output_final, index=iter_, epoch=iter_, time='test_' + self.runtime)
                # pdb.set_trace()
                # log_vars = output_final['log_vars']
                # Update the learning rate
                # self.model.update_learning_rate()
                # Print the losses
                # self.logger.accumulate_losses(log_vars)
                # if iter_ % self.cfg.LOG.STEP_PRINT_LOSS == 0 and comm.is_main_process():
                #     self.logger.print_losses(iter_, )
                #     self.logger.reset()
                #     torch.cuda.empty_cache()
                iter_ += 1
        # save surfels
        self.save_points(0, extra_name=self.cfg.MODEL.RASTERIZER.SAVE_SURFEL_NAME)

    # def render_from_global_model(self, save_to_local=True, **kwargs):
    #     end_iter = kwargs.get("end_iter", 10000)
    #     begin_iter = kwargs.get("begin_iter", 0)
    #     self.model.eval()
    #     valid_psnr_list = []
    #     with torch.no_grad():
    #         iter_ = 0
    #         for input_data in tqdm(self.data_loader):
    #             if iter_ < begin_iter:
    #                 continue
    #             if iter_ > end_iter:
    #                 break
    #             output_final = self.model.forward(input_data, func_name='render_from_global_model', **kwargs)
    #             self.save_image(output_final, index=iter_, epoch=iter_, time='test_' + self.runtime)
    #             # pdb.set_trace()
    #             log_vars = output_final['log_vars']
    #             self.logger.accumulate_losses(log_vars)
    #             self.logger.print_losses(iter_, )
    #             #
    #             # valid_psnr_list.append(output_final.get('psnr_valid_pixel', '-1000'))
    #             iter_ += 1
    #         # mean_psrn_valid = sum(valid_psnr_list)*1.0/len(valid_psnr_list)
    #         # print("Average PSNR on valid pixel: {}".format(mean_psrn_valid))

    def render_from_global_model_metrics(self, save_to_local=True, results_output_name=None, **kwargs):
        end_iter = kwargs.get("end_iter", 10000)
        begin_iter = kwargs.get("begin_iter", 0)
        metrics = {'psnr': [], 'ssim': [], 'lpips_alex': [], 'lpips_vgg': [], 'ssim_1':[]}
        self.model.eval()
        valid_psnr_list = []
        lpips_alex = LPIPS(net='alex', version='0.1')
        lpips_vgg = LPIPS(net='vgg', version='0.1')
        with torch.no_grad():
            iter_ = 0
            for input_data in tqdm(self.data_loader):
                if iter_ < begin_iter:
                    continue
                if iter_ > end_iter:
                    break
                output_final = self.model.forward(input_data, func_name='render_from_global_model', **kwargs)
                render_img_np = self.save_image(output_final, index=iter_, epoch=iter_, time='test_' + self.runtime)  # H x w x 3
                gt_img_np = output_final['gt_image'].cpu().numpy()[0]
                # pdb.set_trace()
                log_vars = output_final['log_vars']
                self.logger.accumulate_losses(log_vars)
                self.logger.print_losses(iter_, )
                #
                # valid_psnr_list.append(output_final.get('psnr_valid_pixel', '-1000'))
                metrics_item = compute_metrics(render_img_np, gt_img_np, lpips_alex=lpips_alex, lpips_vgg=lpips_vgg)
                print("Metrics: ", metrics_item)
                metrics['psnr'].append(metrics_item['psnr'])
                metrics['ssim'].append(metrics_item['ssim'])
                metrics['ssim_1'].append(metrics_item['ssim_1'])
                metrics['lpips_alex'].append(metrics_item['lpips_alex'])
                metrics['lpips_vgg'].append(metrics_item['lpips_vgg'])
                iter_ += 1
            mean_psnr = sum(metrics['psnr'])*1.0 / len(metrics['psnr'])
            mean_ssim = sum(metrics['ssim']) * 1.0 / len(metrics['ssim'])
            mean_ssim_1 = sum(metrics['ssim_1']) * 1.0 / len(metrics['ssim_1'])
            mean_lpips_alex = sum(metrics['lpips_alex']) * 1.0 / len(metrics['lpips_alex'])
            mean_lpips_vgg = sum(metrics['lpips_vgg']) * 1.0 / len(metrics['lpips_vgg'])
            print("Average PSNR : {}".format(mean_psnr))
            print("Average SSIM : {}".format(mean_ssim))
            print("Average SSIM data_range=1.0: {}".format(mean_ssim_1))
            print("Average LPIPS ALEX : {}".format(mean_lpips_alex))
            print("Average LPIPS VGG: {}".format(mean_lpips_vgg))
            if results_output_name is not None:
                with open('./results_{}.txt'.format(results_output_name), 'a') as f:
                    m = "Average PSNR : {}\n".format(mean_psnr)
                    m += "Average SSIM : {}\n".format(mean_ssim)
                    m += "Average SSIM data_range 1.0 : {}\n".format(mean_ssim_1)
                    m += "Average LPIPS ALEX : {}\n".format(mean_lpips_alex)
                    m += "Average LPIPS VGG : {}\n".format(mean_lpips_vgg)
                    f.write(m)



    def test(self, **kwargs):
        # raise NotImplemented()
        self.model.eval()
        self.logger.reset()
        with torch.no_grad():
            for index, input_data in enumerate(self.data_loader):
                start_time = time.time()
                output_final = self.model.forward_render(input_data, no_loss=True, **kwargs)
                print("Time : {:.2f}s".format((time.time() - start_time)))

                self.save_image(output_final, index=index, epoch=index, time='test_' + self.runtime)

    def save_image(self, output, index=0, epoch=0, time='', extra_name=''):
        # gt = output['gt_image'].cpu()
        img = output['rendered_output']['coarse_color']  # tensor
        #
        # gt_img = gt.numpy()
        img = img.cpu().numpy()
        # pdb.set_trace()
        img = img.reshape((self.cfg.DATA.IMG_WH[1], self.cfg.DATA.IMG_WH[0], 3))
        # pdb.set_trace()

        if not os.path.exists(os.path.join(self.log_path, 'images', time,)):
            os.mkdir(os.path.join(self.log_path, 'images', time,))
        save_path = os.path.join(self.log_path, 'images', time,
                                 "index_{}{}_output.png".format(epoch, extra_name))
        self.logger.save_image(img, save_path)
        # if gt_img.shape[0] > 1:
        #     gt_img = gt_img.reshape((self.cfg.DATA.IMG_WH[1], self.cfg.DATA.IMG_WH[0], 3))
        #     save_path = os.path.join(self.log_path, 'images', time,
        #                          "index_{}_gt.png".format(epoch))
        #     self.logger.save_image(gt_img, save_path)
        print("saving Image to {}".format(save_path))
        return img

    def save_checkpoints(self, iterations, extra_name='', **kwargs):
        checkpoints = OrderedDict()
        if isinstance(self.model, (DistributedDataParallel, )):
            model = self.model.module
        else:
            model = self.model
        #
        checkpoints['weights'] = model.state_dict()
        checkpoints['iteration'] = iterations
        #
        # for the optimizer
        checkpoints['optimizer'] = model.get_optimizer_state_dict()
        checkpoints['scheduler'] = model.get_lr_scheduler_state_dict()
        checkpoints['points_info'] = dict()
        # other parameters
        for key in kwargs.keys():
            checkpoints['points_info'][key] = kwargs[key]


        if not os.path.exists(os.path.join(self.log_path)):
            os.mkdir(self.log_path)
        save_path = os.path.join(self.log_path, "{}_{}_checkpoints.pth".format(iterations, extra_name))
        torch.save(checkpoints, save_path)

    def save_points(self, iterations=0, extra_name=''):
        points = {'points':OrderedDict()}
        if isinstance(self.model, (DistributedDataParallel, )):
            model = self.model.module
        else:
            model = self.model
        pcd_dict = model.rasterizer.return_point_cloud()
        points['points'] = pcd_dict
        #
        if not os.path.exists(os.path.join(self.log_path)):
            os.mkdir(self.log_path)
        save_path = os.path.join(self.log_path, "iter{}_{}_points.pth".format(iterations, extra_name))
        torch.save(points, save_path)
        print("Points are saved in {}".format(save_path))
