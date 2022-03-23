from threading import stack_size
import gym
import os
import cv2
import shutil
import datetime
import pickle
import json
import uuid
import logging
from gym.core import ActionWrapper
import numpy as np
from collections import defaultdict
from typing import Generator
from minerl_patched.herobraine.hero import spaces
from custom_tasks import make_3d_cube, make_plane
import matplotlib.pyplot as plt
from sample_factory.envs.env_wrappers import ObservationWrapper, has_image_observations
import cv2
from collections import OrderedDict
from gridworld.task import Task

logger = logging.getLogger(__file__)
IGLU_ENABLE_LOG = os.environ.get('IGLU_ENABLE_LOG', '')


class Wrapper(gym.Wrapper):
    def stack_actions(self):
        if isinstance(self.env, Wrapper):
            return self.env.stack_actions()

    def wrap_observation(self, obs, reward, done, info):
        if hasattr(self.env, 'wrap_observation'):
            return self.env.wrap_observation(obs, reward, done, info)
        else:
            return obs


class ActionsWrapper(Wrapper):
    def wrap_action(self, action) -> Generator:
        raise NotImplementedError

    def stack_actions(self):
        def gen_actions(action):
            for action in self.wrap_action(action):
                wrapped = None
                if hasattr(self.env, 'stack_actions'):
                    wrapped = self.env.stack_actions()
                if wrapped is not None:
                    yield from wrapped(action)
                else:
                    yield action

        return gen_actions

    def step(self, action):
        total_reward = 0
        for a in self.wrap_action(action):
            obs, reward, done, info = super().step(a)
            total_reward += reward
            if done:
                return obs, total_reward, done, info
        return obs, total_reward, done, info


class ObsWrapper(Wrapper):
    def observation(self, obs, reward=None, done=None, info=None):
        raise NotImplementedError

    def wrap_observation(self, obs, reward, done, info):
        new_obs = self.observation(obs, reward, done, info)
        return self.env.wrap_observation(new_obs, reward, done, info)

    def reset(self):
        return self.observation(super().reset())

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['grid'] = obs['grid']
        info['agentPos'] = obs['agentPos']
       # info['obs'] = obs['obs']
        return self.observation(obs, reward, done, info), reward, done, info


class TimeLimit(Wrapper):
    def __init__(self, env, limit):
        super().__init__(env)
        self.limit = limit
        self.step_no = 0

    def reset(self):
        self.step_no = 0
        return super().reset()

    def step(self, action):
        self.step_no += 1
        obs, reward, done, info = super().step(action)
        if self.step_no >= self.limit:
            done = True
        return obs, reward, done, info


class SweeperReward(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_step_garbage_min = 1000
        self.last_step_garbage_max = 0

    def reset(self):
        self.last_step_garbage_min = 1000
        self.last_step_garbage_max = 0
        self.last_step_garbage = 0
        return super().reset()

    def garbage(self, info):
        roi = info['grid'][info['target_grid'] == 0]
        return len((np.where(roi != 0)[0]))

    def calc_reward(self, info):
        garbage = self.garbage(info)
        if garbage > self.last_step_garbage:
            return -0.001
        elif garbage < self.last_step_garbage:
            return 0.001
        self.last_step_garbage = garbage
        return 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        add_reward = self.calc_reward(info)
        reward += add_reward
        return obs, reward, done, info


class RandomTarget(gym.Wrapper):
    def __init__(self, env, random_type = 1, thresh=0.1):
        super().__init__(env)
        self.thresh = thresh
        self.total_reward = 0
        self.sum = self.thresh / 10
        self.count = 0
        self.random_type = random_type
        self.changes = 0

    def reset(self):
        if self.random_type == 1:
            task = make_3d_cube(rand=True)
            self.env.task = Task("", task)
        return super().reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.random_type == 0:
            if done:
                self.count += 1
                self.sum += reward
                self.total_reward = self.sum / self.count
                if (self.total_reward > self.thresh):
                    self.changes += 1
                    task = make_3d_cube(rand=True)
                    self.env.task = task
                    self.sum = self.thresh / 10
                    self.count = 0
                    self.total_reward = self.thresh / 10
                    info['new_env'] = True
        return obs, reward, done, info


class RangetReward (Wrapper):
    def __init__(self, env, rspec=15):
        super().__init__(env)
        self.rspec = rspec

    def calc_reward(self, dist):
        IN = [1, 0.25, 0.05, 0.001, -0.0001, -0.001, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09]
        return IN[int(dist)]

    def blocks_count(self, info):
        return np.sum(info['grid'] != 0)

    def check_goal_closeness(self, info):
        roi = np.where(info['target_grid'] != 0)  # y x z
        goal = np.mean(roi[1]), np.mean(roi[2])
        broi = np.where(info['grid'] != 0)  # y x z
        builds = np.mean(broi[1]), np.mean(broi[2])
        dist = ((goal[0] - builds[0]) ** 2 + (goal[1] - builds[1]) ** 2) ** 0.5
        return self.calc_reward(dist)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = 0
        if self.blocks_count(info) == 1:
            reward = self.check_goal_closeness(info)
            done = True
        return obs, reward, done, info


class CompleteReward(Wrapper):
    def __init__(self, env, spec="hardany", hard_reset=False):
        super().__init__(env)
        self.T = spec
        self.old_bs = 0

    def reset(self):
        self.old_bs = 0
        return super().reset()

    def garbage(self, info):
        roi = info['grid'][info['target_grid'] == 0]
        return len((np.where(roi != 0)[0]))

    def check_complete(self, info):
        roi = info['grid'][info['target_grid'] != 0]
        build_size = len(np.where(roi != 0)[0])
        if self.T == "all":
            return len(np.where(roi == 0)[0]) == 0, build_size
        elif self.T == "any":
            return len(np.where(roi != 0)[0]) > 0, build_size
        elif self.T == "hardany":
            return (len(np.where(roi != 0)[0]) > 0) and (self.garbage(info) == 0), build_size
        elif self.T == "hardall":
            return (len(np.where(roi == 0)[0]) == 0) and (self.garbage(info) == 0), build_size

    def step(self, action):
        obs, reward, done, info = super().step(action)
        check_sr, bs = self.check_complete(info)
        if check_sr:
            reward = 1
            done = True
        else:
            if bs > self.old_bs:
                target_size = np.sum(info['target_grid'] != 0)
                reward = 1 / target_size
                self.old_bs = bs
            else:
                reward = 0
        return obs, reward, done, info


class Closeness(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.dist = 1000000

    def reset(self):
        self.dist = 1000000
        return super().reset()

    def closeness(self, info):
        roi = np.where(info['target_grid'] != 0)  # y x z
        goal = np.mean(roi[1]), np.mean(roi[2])
        agent = info['agentPos'][:3]
        agent_pos = agent[0] + 5, agent[2] + 5

        dist = ((goal[0] - agent_pos[0]) ** 2 + (goal[1] - agent_pos[1]) ** 2) ** 0.5
        return dist

    def calc_reward(self, info):
        d2 = self.closeness(info)
        if d2 < self.dist:
            self.dist = d2
            return 0.001
        elif d2 > self.dist:
            # self.dist = 0
            return 0
        else:
            return 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        add_reward = self.calc_reward(info)
        reward += add_reward
        return obs, reward, done, info


class ClosenessTL(Closeness):
    def __init__(self, env):
        super().__init__(env)
        self.max_steps = 128
        self.steps = 0

    def reset(self):
        self.steps = 0
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.steps += 1
        if self.steps >= self.max_steps:
            reward = -self.closeness(info)
            done = True
        else:
            reward = 0
        return obs, reward/16, done, info


class CompleteScold(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return super().reset()

    def check_filling(self, info):
        roi = info['grid'][info['target_grid'] == 0]
        return len(np.where(roi != 0)[0]) > 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        check_fill = self.check_filling(info)
        if check_fill:
            reward -= 0.005
            done = True
        return obs, reward, done, info


class SizeReward(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.size = 0

    def reset(self):
        self.size = 0
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        intersection = self.env.unwrapped.task.task_monitor.max_int
        reward = max(intersection, self.size) - self.size
        self.size = max(intersection, self.size)
        return obs, reward, done, info


class SelectAndPlace(ActionsWrapper):
    def wrap_action(self, action):
        if action['hotbar'] != 0:
            yield action
            action['use'] = 1
        if action['use'] == 1 or action['attack'] == 1:
            for _ in range(3):
                yield action
        yield action


class RandomRotation(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.vec = np.random.choice([1, -1])
        self.total_rots = np.random.choice(list(range(0,72,5)))

    def reset(self):
        self.steps = 0
        self.vec = np.random.choice([1, -1])
        self.total_rots = np.random.choice(list(range(0,72,5)))
        return super().reset()

    def step(self, action):
        self.steps += 1
        if self.steps <= self.total_rots:
            vec = np.random.choice([1,-1])
            if self.vec == 1:
                 obs, reward, done, info = super().step(7)
            else:
                 obs, reward, done, info = super().step(8)
        else:
             obs, reward, done, info = super().step(action)
        return obs, reward, done, info


def flat_action_space(action_space):
    if action_space == 'human-level':
        return flat_human_level
    if action_space == 'discrete':
        return flat_discrete

def no_op():
    return OrderedDict([('attack', 0), ('back', 0), ('camera', np.array([0., 0.])),
                        ('forward', 0), ('hotbar', 0), ('jump', 0), ('left', 0), ('right', 0),
                        ('use', 0)])

def flat_human_level(env, camera_delta=5):
  #  print(help(env.action_space))
    binary = ['attack', 'forward', 'back', 'left', 'right', 'jump']
    discretes = [no_op()]
    for op in binary:
        dummy = no_op()
        dummy[op] = 1
        discretes.append(dummy)
    camera_x = no_op()
    camera_x['camera'][0] = camera_delta
    discretes.append(camera_x)
    camera_x = no_op()
    camera_x['camera'][0] = -camera_delta
    discretes.append(camera_x)
    camera_y = no_op()
    camera_y['camera'][1] = camera_delta
    discretes.append(camera_y)
    camera_y = no_op()
    camera_y['camera'][1] = -camera_delta
    discretes.append(camera_y)
    for i in range(6):
        dummy = no_op()
        dummy['hotbar'] = i + 1
        discretes.append(dummy)
    discretes.append(no_op())
    return discretes


def flat_discrete(env, camera_delta=5):
    discretes = [env.action_space.no_op()]

    forward = env.action_space.no_op()
    forward['forward'] = 2
    discretes.append(forward)
    backward = env.action_space.no_op()
    backward['forward'] = 1
    discretes.append(backward)

    left = env.action_space.no_op()
    left['strafe'] = 1
    discretes.append(left)
    right = env.action_space.no_op()
    right['strafe'] = 2
    discretes.append(right)

    jumpforward = env.action_space.no_op()
    jumpforward['forward'] = 2
    jumpforward['jump'] = 1
    discretes.append(jumpforward)
    jumpbackward = env.action_space.no_op()
    jumpbackward['forward'] = 1
    jumpbackward['jump'] = 1
    discretes.append(jumpbackward)

    jumpleft = env.action_space.no_op()
    jumpleft['strafe'] = 1
    jumpleft['jump'] = 1
    discretes.append(jumpleft)
    jumpright = env.action_space.no_op()
    jumpright['strafe'] = 2
    jumpright['jump'] = 1
    discretes.append(jumpright)

    attack = env.action_space.no_op()
    attack['attack'] = 1
    discretes.append(attack)

    camera_x = env.action_space.no_op()
    camera_x['camera'][0] = camera_delta
    discretes.append(camera_x)
    camera_x = env.action_space.no_op()
    camera_x['camera'][0] = -camera_delta
    discretes.append(camera_x)
    camera_y = env.action_space.no_op()
    camera_y['camera'][1] = camera_delta
    discretes.append(camera_y)
    camera_y = env.action_space.no_op()
    camera_y['camera'][1] = -camera_delta
    discretes.append(camera_y)
    for i in range(6):
        dummy = env.action_space.no_op()
        dummy['hotbar'] = i + 1
        discretes.append(dummy)
    return discretes

class Discretization(ActionsWrapper):
    def __init__(self, env, flatten):
        super().__init__(env)
        camera_delta = 5
      #  print(env.action_space.no_op)
        self.discretes = flatten(env, camera_delta)
        self.action_space = gym.spaces.Discrete(len(self.discretes))
        self.old_action_space = env.action_space
        self.last_action = None

    def wrap_action(self, action=None, raw_action=None):
        if action is not None:
           # raise Exception(action)
            action = self.discretes[action]
        elif raw_action is not None:
            action = raw_action
        yield action


class FakeIglu(gym.Env):
    def __init__(self, config, wrap_actions=True):
        action_space = config.get('action_space')
        visual = config.get('visual')
        visual_type = config.get('visual_type')
        if visual_type == 'target_grid':
            self.target_grid = True
        if action_space == 'human-level':
            self.action_space = spaces.Dict({
                'forward': spaces.Discrete(2),
                'back': spaces.Discrete(2),
                'left': spaces.Discrete(2),
                'right': spaces.Discrete(2),
                'jump': spaces.Discrete(2),
                'camera': spaces.Box(low=-180.0, high=180.0, shape=(2,)),
                'attack': spaces.Discrete(2),
                'use': spaces.Discrete(2),
                'hotbar': spaces.Discrete(7),
            })
        elif action_space == 'discrete':
            self.action_space = spaces.Dict({
                'move': spaces.Discrete(3),
                'strafe': spaces.Discrete(3),
                'jump': spaces.Discrete(2),
                'camera': spaces.Box(low=-180.0, high=180.0, shape=(2,)),
                'attack': spaces.Discrete(2),
                'use': spaces.Discrete(2),
                'hotbar': spaces.Discrete(7),
            })
        elif action_space == 'continuous':
            self.action_space = spaces.Dict({
                'move_x': spaces.Box(low=-1., high=1., shape=(), dtype=np.float32),
                'move_y': spaces.Box(low=-1., high=1., shape=(), dtype=np.float32),
                'move_z': spaces.Box(low=-1., high=1., shape=(), dtype=np.float32),
                'camera': spaces.Box(low=-180.0, high=180.0, shape=(2,)),
                'attack': spaces.Discrete(2),
                'use': spaces.Discrete(2),
                'hotbar': spaces.Discrete(7),
            })
        if wrap_actions:
            flatten_actions = flat_action_space(action_space)
            self.discrete = flatten_actions(self, camera_delta=5)
            self.full_action_space = self.action_space
            self.action_space = spaces.Discrete(len(self.discrete))
        if visual:
            obs = {
                'pov': spaces.Box(0, 255, (64, 64, 3), dtype=np.float32),
                'inventory': spaces.Box(low=0, high=20, shape=(6,), dtype=np.float32),
                'compass': spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
            }
            if self.target_grid:
                obs['target_grid'] = gym.spaces.Box(low=0.0, high=6.0, shape=(9, 11, 11))
            self.observation_space = spaces.Dict(obs)

        else:
            self.observation_space = spaces.Dict({
                'agentPos': gym.spaces.Box(low=-5000.0, high=5000.0, shape=(5,)),
                'grid': gym.spaces.Box(low=0.0, high=6.0, shape=(9, 11, 11)),
                'inventory': gym.spaces.Box(low=0.0, high=20.0, shape=(6,)),
                'target_grid': gym.spaces.Box(low=0.0, high=6.0, shape=(9, 11, 11))
            })
        self.step = 0

    def reset(self):
        self.step = 0
        return self.observation_space.sample()

    def step(self, action):
        self.step += 1
        done = self.step >= 1000
        reward = 0
        info = {}
        return self.observation_space.sample(), reward, done, info

    def update_taskset(self, *args, **kwargs):
        pass

    def set_task(self, *args, **kwargs):
        pass


class ActLogger(Wrapper):
    def __init__(self, env, every=50):
        super().__init__(env)
        runtime = timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.dirname = f'pics_logs/run-{runtime}'
        self.every = every
        self.filename = None
        self.pic = np.zeros((11, 11, 3))
        self.running_reward = 0
        self.actions = []
        self.info = None
        self.flushed = False
        os.makedirs(self.dirname, exist_ok=True)

    def flush(self):
        if self.filename is not None:
            self.pic[np.where(self.info['grid'] != 0)[1:]] = 0.4  # застройка
            if self.first_cube is not None:
                self.pic[self.first_cube] = 1  # первый кубик
            pic = cv2.resize(self.pic.copy(), (256, 256))
            # print(self.first_cube)
            plt.imsave(self.filename + ".png", pic)
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.flushed = True
        self.first_cube = None
        self.info = None
        uid = str(uuid.uuid4().hex)
        name = f'episode-{timestamp}-{uid}'
        self.filename = os.path.join(self.dirname, name)
        self.pic = np.zeros((11, 11, 3))

    def reset(self):
        if not self.flushed:
            self.flush()
        return super().reset()

    def step(self, action):
        # assuming dict
        self.flushed = False
        obs, reward, done, info = super().step(action)
        if self.info is None:
            self.pic[np.where(obs['target_grid'] != 0)[1:]] = 0.18  # цель
        self.info = info
        pose = info['agentPos'][:3:2] + 5

        if (pose[0] > 0 and pose[0] <= 10) and (pose[1] > 0 and pose[1] <= 10):
            self.pic[int(pose[0]), int(pose[1]), 1] = 0.6  # маршрут
        #  print("go")
        if len(np.where(info['grid'] != 0)[0]) == 1:
            self.first_cube = np.where(info['grid'] != 0)[1:]

            self.pic[np.where(obs['target_grid'] != 0)[1:]] = 0.18  # цель
            # self.pic[first_cube,0] = 10 #первый кубик
        return obs, reward, done, info


class VideoLogger(Wrapper):
    def __init__(self, env, every=50):
        super().__init__(env)
        runtime = timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.dirname = f'action_logs/run-{runtime}'
        self.every = every
        self.filename = None
        self.running_reward = 0
        self.actions = []
        self.flushed = False
        os.makedirs(self.dirname, exist_ok=True)

    def flush(self):
        if self.filename is not None:
            self.out.release()
            with open(f'{self.filename}-obs.pkl', 'wb') as f:
                pickle.dump(self.obs, f)
            self.obs = []
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        uid = str(uuid.uuid4().hex)
        name = f'episode-{timestamp}-{uid}'
        self.filename = os.path.join(self.dirname, name)
        self.running_reward = 0
        self.flushed = True
        self.actions = []
        self.frames = []
        self.obs = []
        self.out = cv2.VideoWriter(f'{self.filename}.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                   20, (64, 64))

    def reset(self):
        if not self.flushed:
            self.flush()
        return super().reset()

    def close(self):
        if not self.flushed:
            self.flush()
        return super().close()

    def step(self, action):
        # assuming dict
        self.flushed = False
        obs, reward, done, info = super().step(action)
        self.actions.append(action)
        if 'obs' in obs:
            image = np.transpose(obs['obs'], (1, 2, 0))
        elif 'obs' in info:
           # print(info['obs'])
            #image = np.transpose(info['obs'], (1, 2, 0))
            image = info['obs']*255
        # print(image.shape)
        self.out.write(image.astype(np.uint8))
        self.obs.append({k: v for k, v in obs.items() if k != 'obs'})
        self.obs[-1]['reward'] = reward
        self.running_reward += reward
        # if done:
        #   self.out.release()
        return obs, reward, done, info


class Logger(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        runtime = timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self.dirname = f'action_logs/run-{runtime}'
        self.filename = None
        self.running_reward = 0
        self.actions = []
        self.flushed = False
        os.makedirs(self.dirname, exist_ok=True)

    def flush(self):
        if self.filename is not None:
            with open(f'{self.filename}-act.pkl', 'wb') as f:
                pickle.dump(self.actions, f)
            with open(f'{self.filename}-obs.pkl', 'wb') as f:
                pickle.dump(self.obs, f)
            self.obs = []
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        uid = str(uuid.uuid4().hex)
        name = f'episode-{timestamp}-{uid}'
        self.filename = os.path.join(self.dirname, name)
        self.running_reward = 0
        self.flushed = True
        self.actions = []
        self.frames = []
        self.obs = []

    def reset(self):
        if not self.flushed:
            self.flush()
        return super().reset()

    def close(self):
        if not self.flushed:
            self.flush()
        return super().close()

    def step(self, action):
        # assuming dict
        self.flushed = False
        obs, reward, done, info = super().step(action)
        self.actions.append(action)
        self.obs.append({k: v for k, v in obs.items() if k != 'pov'})
        self.obs[-1]['reward'] = reward
        self.running_reward += reward
        return obs, reward, done, info

class VisualObservationWrapper(ObsWrapper):
    def __init__(self, env, include_target=False):
        super().__init__(env)
        self.observation_space = {
            'obs': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3)),
            'inventory': gym.spaces.Box(low=0.0, high=20.0, shape=(6,)),
            # 'compass': gym.spaces.Box(low=-180.0, high=180.0, shape=(1,))
        }
        self.include_target = include_target
        if include_target:
            self.observation_space['target_grid'] = \
                gym.spaces.Box(low=0, high=6, shape=(9, 11, 11))
            self.observation_space['agentPos'] = \
                gym.spaces.Box(low=-5000.0, high=5000.0, shape=(5,))
        self.observation_space = gym.spaces.Dict(self.observation_space)

    def observation(self, obs, reward=None, done=None, info=None):
        if info is not None:

            if 'target_grid' in info:
                target_grid = info['target_grid']
                # del info['target_grid']
            else:
                logger.error(f'info: {info}')
                if hasattr(self.unwrapped, 'should_reset'):
                    self.unwrapped.should_reset(True)
                target_grid = self.env.task.target_grid

        else:
             target_grid = self.env.task.target_grid
        observe = {
            'obs': obs['obs'].astype(np.float32),
            'inventory': obs['inventory'],
            # 'compass': np.array([obs['compass']['angle'].item()])
        }
        if self.include_target:
            observe['target_grid'] = target_grid
            observe['agentPos'] = obs['agentPos']
        return observe


class VectorObservationWrapper(ObsWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            'agentPos': gym.spaces.Box(low=-5000.0, high=5000.0, shape=(5,)),
           # 'grid': gym.spaces.Box(low=0.0, high=6.0, shape=(9, 11, 11)),
            'inventory': gym.spaces.Box(low=0.0, high=20.0, shape=(6,)),
            'target_grid': gym.spaces.Box(low=0.0, high=6.0, shape=(9, 11, 11))
        })

    def observation(self, obs, reward=None, done=None, info=None):
        if IGLU_ENABLE_LOG == '1':
            self.check_component(
                obs['agentPos'], 'agentPos', self.observation_space['agentPos'].low,
                self.observation_space['agentPos'].high
            )
            self.check_component(
                obs['inventory'], 'inventory', self.observation_space['inventory'].low,
                self.observation_space['inventory'].high
            )
            self.check_component(
                obs['grid'], 'grid', self.observation_space['grid'].low,
                self.observation_space['grid'].high
            )
        if info is not None:
            if 'target_grid' in info:
                target_grid = self.env.task.target_grid #info['target_grid']
               # del info['target_grid']
            else:
                logger.error(f'info: {info}')
                if hasattr(self.unwrapped, 'should_reset'):
                    self.unwrapped.should_reset(True)
                target_grid = self.env.task.target_grid
        else:
            target_grid = self.env.task.target_grid
        return {
            'agentPos': obs['agentPos'],
            'inventory': obs['inventory'],
       #     'grid': obs['grid'],
            'target_grid': target_grid
        }

    def check_component(self, arr, name, low, hi):
        if (arr < low).any():
            logger.info(f'{name} is below level {low}:')
            logger.info((arr < low).nonzero())
            logger.info(arr[arr < low])
        if (arr > hi).any():
            logger.info(f'{name} is above level {hi}:')
            logger.info((arr > hi).nonzero())
            logger.info(arr[arr > hi])



class MinerlOnlyObs(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']


class PovToObs(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        new_obs_space = self.env.observation_space.spaces
        new_obs_space['obs'] = new_obs_space.pop('pov')
        self.observation_space = gym.spaces.Dict(new_obs_space)

    def observation(self, observation):
        new_obs = observation
        new_obs['obs'] = new_obs.pop('pov')
        return new_obs


class ObsToOneDict(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({'obs': gym.spaces.Dict(self.env.observation_space.spaces)})

    def observation(self, observation):
        return {'obs': observation}


class PixelFormatChwWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            #   raise Exception(env.observation_space.spaces)
            img_obs_space = env.observation_space['obs']
            self.dict_obs_space = True
        else:
            img_obs_space = env.observation_space
            self.dict_obs_space = False

        if not has_image_observations(img_obs_space):
            raise Exception('Pixel format wrapper only works with image-based envs')

        obs_shape = img_obs_space.shape
        max_num_img_channels = 4

        if len(obs_shape) <= 2:
            raise Exception('Env obs do not have channel dimension?')

        if obs_shape[0] <= max_num_img_channels:
            raise Exception('Env obs already in CHW format?')

        h, w, c = obs_shape
        low, high = img_obs_space.low.flat[0], img_obs_space.high.flat[0]
        new_shape = [c, h, w]

        if self.dict_obs_space:
            dtype = env.observation_space.spaces['obs'].dtype if env.observation_space.spaces[
                                                                     'obs'].dtype is not None else np.float32
        else:
            dtype = env.observation_space.dtype if env.observation_space.dtype is not None else np.float32

        new_img_obs_space = spaces.Box(low, high, shape=new_shape, dtype=dtype)

        if self.dict_obs_space:
            self.observation_space = env.observation_space
            self.observation_space.spaces['obs'] = new_img_obs_space
        else:
            self.observation_space = new_img_obs_space

        self.action_space = env.action_space

    @staticmethod
    def _transpose(obs):
        return np.transpose(obs, (2, 0, 1))  # HWC to CHW for PyTorch

    def observation(self, observation):
        if observation is None:
            return observation

        if self.dict_obs_space:
            observation['pov'] = self._transpose(observation['pov'])
        else:
            observation = self._transpose(observation)
        return observation
