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
    RangetReward, \
    Logger
from custom_tasks import make_3d_cube, make_plane
import numpy as np
from gridworld.env import GridWorld
from fast_iglu import IgluFast


def make_iglu(*args, **kwargs):
    custom_grid = make_3d_cube(rand=True)
    env = GridWorld(custom_grid, render=False)

    env = SelectAndPlace(env)

    env = VectorObservationWrapper(env)

    env = TimeLimit(env, limit=750)
    env = RangetReward(env)

    env = Discretization(env, flat_action_space('human-level'))
    #env = RandomRotation(env)
    env = RandomTarget(env)

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
