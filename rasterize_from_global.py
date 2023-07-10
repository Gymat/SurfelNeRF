import sys
import os
import pathlib
import copy
import torch
import numpy as np
import time
np.random.seed(0)
import pdb

from config.setup import setup_config, setup_argparser
from engine.train_loop_surfel import Train_loop_online
from ddp_api.launch import launch
from ddp_api import comm
from models.pipelines.build import build_model
from torch.nn.parallel import DistributedDataParallel
from data.build import build_dataloader
from models.checkpoints.checker import CheckPointer


def main(cfg):
    torch.backends.cudnn.benchmark = True
    ngpus_per_node = torch.cuda.device_count()
    assert ngpus_per_node >= args.num_gpus
    print("ngpus_per_node: %s" % ngpus_per_node)
    torch.manual_seed(0)
    #
    launch(
        do_engine,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(config,),
    )


def do_engine(config, **kwargs):
    model = build_model(config)
    # The .train() need to be checked very very carefully!!!
    # Since there are BNs in some frozen modules, such as perceptual network(vgg or resnet), it may be better
    #  to re-write this .train() for the module which needs grad (require_grad = True).
    # model.train()

    # Whether to load the checkpoints
    model.to_gpu()

    checkpointer = CheckPointer(config, model)
    # model.train()


    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            checkpointer.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_render(config, model, distributed, iteration=checkpointer.iter_)


def do_render(config, model, distributed, **kwargs):
    data_loader = build_dataloader(config, distributed=distributed)

    train_loop = Train_loop_online(config, model, data_loader, **kwargs)

    train_loop.render_from_global_model_metrics(results_output_name=config.TXT_NAME, **kwargs)


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    print("Command Line Args:", args)
    config = setup_config(args)
    print("Call with args:")
    print(config)

    main(config)
