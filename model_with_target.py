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

from torch import nn
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor, get_preprocessor

torch, nn = try_import_torch()

CUDA_LAUNCH_BLOCKING = 1


class ResudualBlock(nn.Module):
    def __init__(self, Nin, out, ksize=3, stride=1):
        super(ResudualBlock, self).__init__()
        self.conv1 = nn.Conv2d(Nin, out, ksize, stride, padding=1)
        self.Relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out)

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.bn(x)
        x = self.Relu(x)

        x = self.conv1(x)
        x = self.bn(x)

        x = x + input
        output = self.Relu(x)
        return output


class LargePovBaselineModelTarget(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs,
                         model_config, name)
        if num_outputs is None:
            # required by rllib's lstm wrapper
            num_outputs = int(np.product(self.obs_space.shape))
        pov_embed_size = 128
        inv_emded_size = 128
        target_emded_size = 128
        embed_size = 128 * 2

        self.conv3x3 = nn.Conv2d(3, 64, 3, stride=1)
        self.relu1 = nn.ReLU()
        self.conv3x3_2 = nn.Conv2d(64, 128, 3, stride=1)
        self.max_pull = nn.MaxPool2d((3, 3), stride=2)

        self.res_block_1 = ResudualBlock(128, 128)
        self.res_block_2 = ResudualBlock(128, pov_embed_size)

        self.inventory_compass_emb = nn.Sequential(
            nn.Linear(7, inv_emded_size),
            nn.ReLU(),
            nn.Linear(inv_emded_size, inv_emded_size),
            nn.ReLU(),
        )
        self.target_grid_emb = nn.Sequential(
            nn.Linear(9*11*11, target_emded_size),
            nn.ReLU(),
            nn.Linear(target_emded_size, target_emded_size),
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
        pov_embed = self.conv3x3(pov)
        pov_embed = self.relu1(pov_embed)
        pov_embed = self.conv3x3_2(pov_embed)
        pov_embed = self.max_pull(pov_embed)

        pov_embed = self.res_block_1(pov_embed)
        pov_embed = self.res_block_2(pov_embed)

        pov_embed = pov_embed.mean(axis=2)
        pov_embed = pov_embed.mean(axis=2)
        pov_embed = pov_embed.reshape(pov_embed.shape[0], -1)

        tg = obs['target_grid']/6
        tg = tg.reshape(tg.shape[0], -1)
        tg_embed = self.target_grid_emb(tg)

        inventory_compass = torch.cat([obs['inventory'], obs['compass']], 1)
        inv_comp_emb = self.inventory_compass_emb(inventory_compass)

        head_input = torch.cat([pov_embed, inv_comp_emb,tg_embed], 1)

        return self.head(head_input), state




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