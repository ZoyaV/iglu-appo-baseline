import gridworld
import gym
from gridworld.env import GridWorld
import numpy as np
import random

from minerl_patched.herobraine.hero import spaces
import pyglet

pyglet.options["headless"] = True
from gridworld.world import World
from gridworld.control import Agent
from gridworld.render import Renderer, setup
from gridworld.task import Task

from gym.spaces import Dict, Box, Discrete, Tuple
from gym import Env
import numpy as np
from copy import copy
import os
from collections import OrderedDict


class gwDict(Dict):
    def no_op(self):
        return OrderedDict([('attack', 0), ('back', 0), ('camera', np.array([0., 0.])),
                            ('forward', 0), ('hotbar', 0), ('jump', 0), ('left', 0), ('right', 0),
                            ('use', 0)])


class IgluFast(Env):
    def __init__(self, target, render=True) -> None:
        self.world = World()
        self.agent = Agent(self.world, sustain=False)
        self.grid = np.zeros((9, 11, 11), dtype=np.int32)
        self.task = Task('', target)
        self.world.add_callback('on_add', self.add_block)
        self.world.add_callback('on_remove', self.remove_block)
        # self.action_space = Dict({
        #     'forward': Discrete(2),
        #     'back': Discrete(2),
        #     'left': Discrete(2),
        #     'right': Discrete(2),
        #     'jump': Discrete(2),
        #     'attack': Discrete(2),
        #     'use': Discrete(2),
        #     'camera': Box(low=-5, high=5, shape=(2,)),
        #     'hotbar': Discrete(7)
        # })

        self.mapping = {
            'forward': 0,
            'back': 1,
            'left': 2,
            'right': 3,
            'jump': 4,
            'attack': 5,
            'use': 6,
            'camera': 7,
            'hotbar': 8,
        }

        self.action_space = Tuple((
            Discrete(2),  # forward
            Discrete(2),  # back
            Discrete(2),  # left
            Discrete(2),  # right
            Discrete(2),  # jump
            Discrete(2),  # attack
            Discrete(2),  # use
            Discrete(5),  # camera
            Discrete(7)  # hotbar
        ))

        self.inv_mapping = ['forward', 'back', 'left', 'right', 'jump', 'attack', 'use'] \
                           + ['camera']*4 + ['hotbar']*6

        self.observation_space = Dict({
            'agentPos': Box(low=0, high=360, shape=(5,)),
            'inventory': Box(low=0, high=20, shape=(6,)),
      #      'obs': Box(low=0, high=1, shape=(64, 64, 3)),
        })
        self.max_int = 0
        self.do_render = render
        if render:
            self.renderer = Renderer(self.world, self.agent,
                                     width=64, height=64, caption='Pyglet', resizable=False)
            setup()
        else:
            self.renderer = None

    def add_block(self, position, kind):
        if self.world.initialized:
            x, y, z = position
            x += 5
            z += 5
            y += 1
            self.grid[y, x, z] = kind

    def remove_block(self, position):
        if self.world.initialized:
            x, y, z = position
            self.grid[y, x, z] = 0

    def get_pov(self):
        return self.renderer.render()

    def reset(self):
        self.prev_grid_size = 0
        self.max_int = 0
        for block in tuple(self.world.placed):
            self.world.remove_block(block)
        self.agent.position = (0, 0, 0)
        self.agent.rotation = (0, 0)
        self.agent.inventory = [20 for _ in range(6)]
        obs = {
            'agentPos': np.array([0, 0, 0, 0, 0]),
            'inventory': np.array([20 for _ in range(6)]),
       #     'obs': self.get_pov()
        }
        return obs

    def render(self, **kwargs):
        if not self.do_render:
            raise ValueError('create env with render=True')
        return self.renderer.render()

    def parse_action_inv(self, action):
        naction = (0,0,0,0,0,0,0,0,0)
        try:
            num = np.where(action!=0)[0][0]
            waction = self.inv_mapping[num]
            if action == 'camera':
                naction[self.mapping[waction]] = action - 7
            if action == 'hotbar':
                naction[self.mapping[waction]] = action - 11
        except:
            return naction
        return naction

    def parse_action(self, action):
        strafe = [0, 0]
        if action[self.mapping['forward']]:
            strafe[0] += -1
        if action[self.mapping['back']]:
            strafe[0] += 1
        if action[self.mapping['left']]:
            strafe[1] += -1
        if action[self.mapping['right']]:
            strafe[1] += 1
        jump = bool(action[self.mapping['jump']])
        if action[self.mapping['hotbar']] == 0:
            inventory = None
        else:
            inventory = action[self.mapping['hotbar']]

        camera = action[self.mapping['camera']]

        remove = bool(action[self.mapping['attack']])
        add = bool(action[self.mapping['use']])
        return strafe, jump, inventory, camera, remove, add

    def step(self, action):

        strafe, jump, inventory, camera, remove, add = self.parse_action(action)
        self.agent.movement(strafe=strafe, jump=jump, inventory=inventory)
        if camera == 0:
            self.agent.move_camera(0, 0)
        elif camera == 1:
            self.agent.move_camera(-5, 0)
        elif camera == 2:
            self.agent.move_camera(0, -5)
        elif camera == 3:
            self.agent.move_camera(5, 0)
        elif camera == 4:
            self.agent.move_camera(0, 5)

        self.agent.place_or_remove_block(remove=remove, place=add)
        self.agent.update(dt=1 / 20.)
        x, y, z = self.agent.position
        pitch, yaw = self.agent.rotation
        obs = {'agentPos': np.array([x, y, z, pitch, yaw])}
        obs['inventory'] = np.array(copy(self.agent.inventory))
        obs['grid'] = self.grid.copy()
       # obs['obs'] = self.get_pov()
        grid_size = (self.grid != 0).sum().item()
        wrong_placement = (self.prev_grid_size - grid_size) * 1
        max_int = self.task.maximal_intersection(self.grid) if wrong_placement != 0 else self.max_int
        done = max_int == self.task.target_size
        self.prev_grid_size = grid_size
        right_placement = (max_int - self.max_int) * 2
        self.max_int = max_int
        if right_placement == 0:
            reward = wrong_placement
        else:
            reward = right_placement
        return obs, reward, done, {'target_grid': self.task.target_grid}