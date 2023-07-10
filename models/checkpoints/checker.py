import torch
import os

import pdb

"""
This provide two functins:
(1) dump the config
(2) load the checkpoints for model, and provide epoch and iterations
"""

class CheckPointer():
    def __init__(self, config, model):
        super(CheckPointer, self).__init__()
        # config
        self.cfg = config
        # common params
        self.epoch = 0
        self.iter_ = 0
        #
        self.model = model

        if not os.path.exists(os.path.join(self.cfg.LOG.CHECKPOINT_PATH, self.cfg.LOG.TASK_NAME)):
            os.makedirs(os.path.join(self.cfg.LOG.CHECKPOINT_PATH, self.cfg.LOG.TASK_NAME), exist_ok=True)
        self.log_path = os.path.join(self.cfg.LOG.CHECKPOINT_PATH, self.cfg.LOG.TASK_NAME)

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
        if self.cfg.MODEL.DEPTH.BUILD_DEPTH:
            self.model.depth_net.load_self_weights_only()


    def load_checkpoints(self, path):
        weights = torch.load(path)
        model_weights = weights['weights']
        if 'optimizer' in weights:
            optimizer_weights = weights['optimizer']
        if 'scheduler' in weights and self.cfg.IS_TRAIN and not self.cfg.TRAIN.RESET_OPTIM:
            self.model.load_scheduler_state_dict(weights['scheduler'])
        #
        with torch.no_grad():
            self.model.load_state_dict(model_weights, strict=False)

        #
        if self.cfg.IS_TRAIN and not self.cfg.TRAIN.RESET_OPTIM and 'optimizer' in weights:
            self.model.load_optimizer_state_dict(optimizer_weights)
        # if 'points' in weights and self.cfg.MODEL.NEURAL_POINTS.LOAD_POINTS:
        #     self.model.load_points(weights['points'])
        return weights

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