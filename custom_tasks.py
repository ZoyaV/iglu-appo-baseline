import gym
import numpy as np
#from iglu.tasks import CustomTasks
import random


def make_3d_cube(rand=False):
    custom_grid = np.zeros((9, 11, 11))  # (y, x, z)
    if rand:
        x = random.randint(0, 10)
        y = random.randint(0, 10)
        # z = random.randint(0, 8)
    else:
        x = 1  # 8
        y = 1  # 7
    z = 0
    print("\n CHANGE TASK \n", x, y, z)
    print(z, x, y)
    custom_grid[z, x, y] = 1  # blue color
    return CustomTasks([
        ('<Architect> Please, build a stack of three blue blocks somewhere.\n'
         '<Builder> Sure.',
         custom_grid)])


def make_plane(rand=False, size=0.35, rand_conf=1 / 4):
    custom_grid = np.zeros((9, 11, 11))
    plane_size = int(size * 11)
    if rand == True:
        if rand_conf == 1:
            x = random.randint(0, 10 - plane_size)
            y = random.randint(0, 10 - plane_size)
        if rand_conf == 1 / 4:
            x = random.choice([0, 11 - plane_size])
            y = random.choice([0, 11 - plane_size])
    else:
        x = 0
        y = 0
    custom_grid[0, x:x + plane_size, y:y + plane_size] = 1  # blue color
    return custom_grid
