from environment import *
from visualizer import *

env = FrozenLake(2, 10, True)
env.reset()
Done = False
while not Done:
    state, reward, Done = env.step(env.action_space_sample())
    env.render()
    if Done:
        env.reset()
