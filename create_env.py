import gym
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper
from sample_factory.envs.env_wrappers import PixelFormatChwWrapper
import minerl
from RayEnvWrapper import WrapperRayVecEnv
import gym
import iglu
import sys
import wandb
from iglu.tasks import RandomTasks, TaskSet
from wrappers import \
    SelectAndPlace, \
    Discretization, \
    flat_action_space, \
    SizeReward, \
    TimeLimit, \
    VectorObservationWrapper, \
    VisualObservationWrapper, \
    VisualOneBlockObservationWrapper, \
    CompleteReward, \
    CompleteScold, \
    Closeness, \
    SweeperReward, \
    RandomTarget, \
    Logger
from custom_tasks import make_3d_cube, make_plane

class MinerlOnlyObs(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']


def make_iglu(*args, **kwargs):
    import iglu
    env = gym.make('IGLUSilentBuilder-v0', max_steps=5000)
    env.update_taskset(make_plane(rand=True))
    env = SelectAndPlace(env)
    env = Discretization(env, flat_action_space('human-level'))
    env = VisualObservationWrapper(env, True)
    env = CompleteReward(env)
    env = TimeLimit(env, limit=500)
    env = Closeness(env)
    env = SweeperReward(env)
    env = RandomTarget(env)

    return env


def make_ray_iglu(num_workers=1, envs_per_worker=1):
    vec_env = WrapperRayVecEnv(make_iglu, num_workers, envs_per_worker)
    vec_env.reward_range = (-float('inf'), float('inf'))
    vec_env.num_agents = num_workers * envs_per_worker
    vec_env.is_multiagent = True
    return vec_env
