import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers.networks import init_seq

def premlp_init(opt, in_channels):
    in_channels = in_channels  # Hard code !!!
    out_channels = opt.MODEL.NEURAL_POINTS.CHANNELS
    blocks = []
    act = nn.LeakyReLU

    for i in range(opt.MODEL.AGGREGATOR.SHADING_FEATURE_MLP_LAYER1):
        blocks.append(nn.Linear(in_channels, out_channels))
        blocks.append(act(inplace=True))
        in_channels = out_channels
    blocks = nn.Sequential(*blocks)
    init_seq(blocks)
    return blocks

