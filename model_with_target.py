from typing import Sequence
import gym
import numpy as np
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.annotations import override

from sample_factory.algorithms.appo.model_utils import register_custom_encoder, EncoderBase, nonlinearity, get_obs_shape
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements

from torch import nn
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor, get_preprocessor

torch, nn = try_import_torch()
import logging

CUDA_LAUNCH_BLOCKING = 1


class ResBlock(nn.Module):
    def __init__(self, cfg, input_ch, output_ch, timing,kernel = 3, wl = True):
        super().__init__()

        self.timing = timing
        if wl:
            layers = [
                nonlinearity(cfg),
                nn.Conv2d(input_ch, output_ch, kernel_size=kernel, stride=1, padding=1),  # padding SAME
                nonlinearity(cfg),
                nn.Conv2d(output_ch, output_ch, kernel_size=kernel, stride=1, padding=1),  # padding SAME
            ]
        else:
            layers = [
                nn.Conv2d(input_ch, output_ch, kernel_size=kernel, stride=1, padding=1),  # padding SAME
                nn.Conv2d(output_ch, output_ch, kernel_size=kernel, stride=1, padding=1),  # padding SAME
            ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x):
        with self.timing.add_time('res_block'):
            identity = x
            out = self.res_block_core(x)
            with self.timing.add_time('res_block_plus'):
                out = out + identity
            return out


class ResnetEncoderWithTarget(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)
        #raise Exception(obs_space)
      #  obs_shape = get_obs_shape(obs_space['obs'])
      #  input_ch = obs_shape.obs[0]

        target_shape = get_obs_shape(obs_space['target_grid'])
        input_ch_targ = target_shape.obs[0]

        inv_emded_size = 64
        resnet_conf = [[32, 2], [64, 2], [64, 2]]
        target_conf = [[32, 3], [64,3]]
       # target_conf = [[16, 2], [32, 2], [32, 2]]


        #curr_input_channels = input_ch
        layers = []
        layers_target = []

        ### OBS embedding
       # for i, (out_channels, res_blocks) in enumerate(resnet_conf):
      #      layers.extend([
      #          nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
      #          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # padding SAME
       #     ])
       #     for j in range(res_blocks):
      #          layers.append(ResBlock(cfg, out_channels, out_channels, self.timing))
      #      curr_input_channels = out_channels
     #   layers.append(nonlinearity(cfg))
     #   self.conv_head = nn.Sequential(*layers)

        ### Target embedding
        kernel = 3
        for i, (out_channels, res_blocks) in enumerate(target_conf):
            layers_target.extend([
                nn.Conv2d(input_ch_targ, out_channels, kernel_size=kernel, stride=1, padding=1),  # padding SAME
            ])
            for j in range(res_blocks):
                layers_target.append(ResBlock(cfg, out_channels, out_channels, self.timing, kernel = kernel, wl = True))
            input_ch_targ = out_channels
        layers_target.append(nonlinearity(cfg))
        self.conv_target = nn.Sequential(*layers_target)

        self.inventory_compass_emb = nn.Sequential(
            nn.Linear(11, inv_emded_size),
            nn.ReLU(),
            nn.Linear(inv_emded_size, inv_emded_size),
            nn.ReLU(),
        )


        self.conv_target_out_size = calc_num_elements(self.conv_target, target_shape.obs)
        self.init_fc_blocks(self.conv_target_out_size+  inv_emded_size)

    #   self.init_fc_blocks(self.conv_head_out_size)

    def forward(self, obs_dict):


       # x = self.conv_head(obs_dict['obs'])
      #  x = x.contiguous().view(-1, self.conv_head_out_size)

        suma = torch.from_numpy(np.array([10,8,10,180,360])).cuda()
        inventory_compass = torch.cat([obs_dict['inventory']/20, (obs_dict['agentPos']+suma)/suma], -1)
        inv_comp_emb = self.inventory_compass_emb(inventory_compass)

        tg = self.conv_target(obs_dict['target_grid']/6)
        tg_embed = tg.contiguous().view(-1, self.conv_target_out_size)

        head_input = torch.cat([ inv_comp_emb, tg_embed], -1)

        x = self.forward_fc_blocks(head_input)
        return x

