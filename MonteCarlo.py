import environment as fl
import numpy as np
import matplotlib.pyplot as plt

env = fl.FrozenLake(1, 4, False)

# Initialize the Q-table with zeros
Q = np.zeros([env.map.size, env.action_space.size])

# Initialize the policy to be a random epsilon-greedy policy
eplison = 0.1
policy = np.ones([env.map.size, env.action_space.size]) * eplison / env.action_space.size
for i in range(env.map.size):
    a_star = np.argmax(Q[i])
    policy[i][a_star] += 1 - eplison

# Set learning parameters
lr = 0.8
num_episodes = 2000

# Initialize lists to contain total rewards and steps p