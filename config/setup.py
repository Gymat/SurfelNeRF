import argparse
import os
import random
import sys
from config.defaults import get_config

def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
                        default="configs/scene0000_01.yaml",
                        metavar="FILE",
                        help="path to config file")
    parser.add_argument(
        "--pkl_path",
        default="configs/scene0000_01.yaml",
        metavar="FILE",
        help="path to config file")
    parser.add_argument(
        "--device",
        default="cuda",
        metavar="FILE",
        help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() + random.randint(0, 100) if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    # parser.add_argument(
    #     "--uv-folder",
    #     help="The folder of uv images used in post-process",
    #     default=None,
    # )
    parser.add_argument(
        "--test_only",
        help="The folder of uv images used in post-process",
        default=False,
    )
    # parser.add_argument(
    #     "--single_ref",
    #     help="The folder of uv images used in post-process",
    #     default=False,
    # )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def setup_config(args=None):
    config = get_config()
    if args is not None:
        config.merge_from_file(args.config_file)
        config.merge_from_list(args.opts)
    config.freeze()
    return config

if __name__ == '__main__':
    cfg = get_config()
    print(cfg)
