import pdb

import torch
import torch.nn as nn
from models.gru.build import FUSION_REGISTRY

@FUSION_REGISTRY.register()
class GRU2D_naive(nn.Module):
    def __init__(self, cfg):
        super(GRU2D_naive, self).__init__()
        input_channel  = cfg.MODEL.FUSION.INPUT_DIM
        hidden_channel = cfg.MODEL.FUSION.HIDDEN_DIM
        weights_dim = cfg.MODEL.FUSION.WEIGHTS_DIM
        self.mlp_z = nn.Linear(hidden_channel + input_channel, hidden_channel)
        self.mlp_r = nn.Linear(hidden_channel + input_channel, hidden_channel)
        self.mlp_n = nn.Linear(hidden_channel + input_channel, hidden_channel)  # for extra input feats

    def forward(self, input_feat, hidden_feat, input_weights_emb, hidden_weights_emb,
                input_dirs, hidden_dirs, **kwargs):
        if hidden_feat is None:
            hidden_feat = torch.zeros_like(input_feat)
        # pdb.set_trace()
        # input_feat_1 = torch.cat((input_feat, input_weights_emb, input_dirs), dim=-1)
        # hidden_feat_1 = torch.cat((hidden_feat, hidden_weights_emb, hidden_dirs), dim=1)
        concat_input = torch.cat((hidden_feat, input_feat), dim=-1)  # B x N x C
        r = torch.sigmoid(self.mlp_r(concat_input))
        z = torch.sigmoid(self.mlp_z(concat_input))
        #
        update_feat = torch.cat((r * hidden_feat, input_feat), dim=-1)
        q = torch.tanh(self.mlp_n(update_feat))
        #
        output = (1 - z) * hidden_feat + z * q
        # pdb.set_trace()
        return output

@FUSION_REGISTRY.register()
class GRU2D_naive_Wweights(nn.Module):
    def __init__(self, cfg):
        super(GRU2D_naive_Wweights, self).__init__()
        input_channel  = cfg.MODEL.FUSION.INPUT_DIM
        hidden_channel = cfg.MODEL.FUSION.HIDDEN_DIM
        weights_dim = cfg.MODEL.FUSION.WEIGHTS_DIM
        self.mlp_z = nn.Linear(hidden_channel + input_channel + 4*weights_dim, hidden_channel)
        self.mlp_r = nn.Linear(hidden_channel + input_channel + 4*weights_dim, hidden_channel)
        self.mlp_n = nn.Linear(hidden_channel + input_channel + 2*weights_dim, hidden_channel)  # for extra input feats

    def forward(self, input_feat, hidden_feat, input_weights_emb, hidden_weights_emb,
                input_dirs, hidden_dirs, **kwargs):
        if len(input_feat.size()) == 2 and input_feat.size(0) == 1:
            input_feat = input_feat.unsqueeze(1)
        if hidden_feat is None:
            hidden_feat = torch.zeros_like(input_feat)
        if hidden_weights_emb is None:
            hidden_weights_emb = torch.zeros_like(input_weights_emb)
        # pdb.set_trace()
        try:
            input_feat_1 = torch.cat((input_feat, input_weights_emb), dim=-1)
            hidden_feat_1 = torch.cat((hidden_feat, hidden_weights_emb), dim=-1)
        except:
            pdb.set_trace()
        concat_input = torch.cat((hidden_feat_1, input_feat_1), dim=-1)  # B x N x C
        r = torch.sigmoid(self.mlp_r(concat_input))
        z = torch.sigmoid(self.mlp_z(concat_input))
        #
        update_feat = torch.cat((r * hidden_feat, input_feat_1), dim=-1)
        q = torch.tanh(self.mlp_n(update_feat))
        #
        output = (1 - z) * hidden_feat + z * q
        # pdb.set_trace()
        return output

