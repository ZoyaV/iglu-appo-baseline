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

from sample_factory.algorithms.appo.model_utils import register_custom_encoder, EncoderBase, nonlinearity,get_obs_shape,ResBlock
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements

from torch import nn
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor, get_preprocessor

torch, nn = try_import_torch()

CUDA_LAUNCH_BLOCKING = 1


class ResnetEncoderWithTarget(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]

        if cfg.encoder_subtype == 'resnet_impala':
            # configuration from the IMPALA paper
            resnet_conf = [[16, 2], [32, 2], [32, 2]]
        else:
            raise NotImplementedError(f'Unknown resnet subtype {cfg.encoder_subtype}')

        curr_input_channels = input_ch

        inv_emded_size = 32
        target_emded_size = 32
        embed_size = 64
        self.inventory_compass_emb = nn.Sequential(
            nn.Linear(7, inv_emded_size),
            nn.ReLU(),
            nn.Linear(inv_emded_size, inv_emded_size),
            nn.ReLU(),
        )
        self.target_grid_emb = nn.Sequential(
            nn.Linear(9 * 11 * 11, target_emded_size),
            nn.ReLU(),
            nn.Linear(target_emded_size, target_emded_size),
            nn.ReLU(),
        )
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend([
                nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # padding SAME
            ])

            for j in range(res_blocks):
                layers.append(ResBlock(cfg, out_channels, out_channels, self.timing))

            curr_input_channels = out_channels

        layers.append(nonlinearity(cfg))

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        self.init_fc_blocks(self.conv_head_out_size)

    def forward(self, obs_dict):
        x = self.conv_head(obs_dict['obs'])
        x = x.contiguous().view(-1, self.conv_head_out_size)

        inventory_compass = torch.cat([obs_dict['inventory'], obs_dict['compass']], -1)
        inv_comp_emb = self.inventory_compass_emb(inventory_compass)

        tg = obs_dict['target_grid']
        tg = tg.reshape(tg.shape[0], -1)
        tg_embed = self.target_grid_emb(tg)

        head_input = torch.cat([x, inv_comp_emb, tg_embed], -1)

        x = self.forward_fc_blocks(head_input)
        return x



class PovBaselineModelTarget(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs,
                         model_config, name)
        if num_outputs is None:
            # required by rllib's lstm wrapper
            num_outputs = int(np.product(self.obs_space.shape))
        pov_embed_size = 256
        inv_emded_size = 256
        embed_size = 512
        target_emded_size = 256
        self.pov_embed = nn.Sequential(
            nn.Conv2d(3, 64, 4, 4),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 4),
            nn.ReLU(),
            nn.Conv2d(128, pov_embed_size, 4, 4),
            nn.ReLU(),
        )
        self.inventory_compass_emb = nn.Sequential(
            nn.Linear(7, inv_emded_size),
            nn.ReLU(),
            nn.Linear(inv_emded_size, inv_emded_size),
            nn.ReLU(),
        )
        self.target_grid_emb = nn.Sequential(
            nn.Linear(9 * 11 * 11, inv_emded_size),
            nn.ReLU(),
            nn.Linear(inv_emded_size, inv_emded_size),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(pov_embed_size + inv_emded_size + target_emded_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, num_outputs),
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        pov = obs['pov'] / 255. - 0.5
        pov = pov.transpose(2, 3).transpose(1, 2).contiguous()
        pov_embed = self.pov_embed(pov)
        pov_embed = pov_embed.reshape(pov_embed.shape[0], -1)

        inventory_compass = torch.cat([obs['inventory'], obs['compass']], -1)
        inv_comp_emb = self.inventory_compass_emb(inventory_compass)


        tg = obs['target_grid']
        tg = tg.reshape(tg.shape[0], -1)
        tg_embed = self.target_grid_emb(tg)

        head_input = torch.cat([pov_embed, inv_comp_emb, tg_embed], -1)
        return self.head(head_input), state