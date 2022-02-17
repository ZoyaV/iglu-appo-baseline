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

class DiscreteBase(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_dict = {}
        self.action_space = gym.spaces.Discrete(len(self.action_dict))

    def step(self, action):
        s, r, done, info = self.env.step(self.action_dict[action])
        return s, r, done, info

    def sample_action(self):
        return self.action_space.sample()


class DiscreteWrapper(DiscreteBase):
    def __init__(self, env, always_attack=True, angle=5):
        super().__init__(env)
        self.action_dict = {
            0: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 1, 'jump': 0, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0},
            1: {'attack': always_attack, 'back': 0, 'camera': [0, angle], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            2: {'attack': 1, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0,
                'sprint': 0},
            3: {'attack': always_attack, 'back': 0, 'camera': [angle, 0], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            4: {'attack': always_attack, 'back': 0, 'camera': [-angle, 0], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            5: {'attack': always_attack, 'back': 0, 'camera': [0, -angle], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            6: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 1, 'jump': 1, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0},
            7: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 1, 'right': 0,
                'sneak': 0, 'sprint': 0},
            8: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 1,
                'sneak': 0, 'sprint': 0},
            9: {'attack': always_attack, 'back': 1, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0}}
        self.action_space = gym.spaces.Discrete(len(self.action_dict))


class MinerlOnlyObs(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']


def make_treechop(*args, **kwargs):
    import iglu
    env = gym.make('IGLUSilentBuilder-v0', max_steps=5000)
    env = DiscreteWrapper(env)
    env = MinerlOnlyObs(env)
    env = PixelFormatChwWrapper(env)

    #env = MultiAgentWrapper(env)

    return env


def make_ray_treechop(num_workers=1, envs_per_worker=1):
    vec_env = WrapperRayVecEnv(make_treechop, num_workers, envs_per_worker)
    vec_env.reward_range = (-float('inf'), float('inf'))
    vec_env.num_agents = num_workers * envs_per_worker
    vec_env.is_multiagent = True
    return vec_env
