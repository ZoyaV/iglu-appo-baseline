import gym
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper
from sample_factory.envs.env_wrappers import ObservationWrapper, has_image_observations
import minerl
from RayEnvWrapper import WrapperRayVecEnv
import gym
from gym import spaces
#import iglu
import sys
import wandb
#from iglu.tasks import RandomTasks, TaskSet
from wrappers import \
    SelectAndPlace, \
    Discretization, \
    DiscretizationTuple, \
    flat_action_space, \
    SizeReward, \
    TimeLimit, \
    VectorObservationWrapper, \
    VisualObservationWrapper, \
    CompleteReward, \
    CompleteScold, \
    Closeness, \
    ClosenessTL, \
    SweeperReward, \
    RandomTarget, \
    VideoLogger, \
    RandomRotation, \
    Logger
from custom_tasks import make_3d_cube, make_plane
import numpy as np
from gridworld.env import GridWorld
from fast_iglu import IgluFast

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
    """TODO? This can be optimized for VizDoom, can we query CHW directly from VizDoom?"""

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


def make_iglu(*args, **kwargs):
    custom_grid = make_plane(rand=True)
    env = GridWorld(custom_grid, render=False)

   # env = SelectAndPlace(env)

    env = VectorObservationWrapper(env)

    # env = CompleteReward(env)
    env = TimeLimit(env, limit=1000)
    env = ClosenessTL(env)
    # env = CompleteScold(env)
    # env = SweeperReward(env)

    env = Discretization(env, flat_action_space('human-level'))
    env = RandomRotation(env)
    #  env = RandomTarget(env)

    num_workers, envs_per_worker = 1, 1
    env.reward_range = (-float('inf'), float('inf'))
    env.num_agents = num_workers * envs_per_worker
    env.is_multiagent = False

    return env


def make_ray_iglu(num_workers=1, envs_per_worker=1):
    #   vec_env =Discretization(make_iglu(), flat_action_space('human-level'))
    vec_env = WrapperRayVecEnv(make_iglu, num_workers, envs_per_worker)
    vec_env.reward_range = (-float('inf'), float('inf'))
    vec_env.num_agents = num_workers * envs_per_worker
    vec_env.is_multiagent = True
    return vec_env
