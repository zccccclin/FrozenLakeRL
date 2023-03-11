import matplotlib.pyplot as plt
import numpy as np

episode = []
eps_list = []
epsilon_initial = 0.1
num_episodes = 20000

epsilon_final = 0
graph = []
epsilon = epsilon_initial
for i in range(1, num_episodes+1):
    episode.append(i)
    eps_list.append(epsilon)
    #epsilon = max(epsilon_final, epsilon_initial - (epsilon_initial - epsilon_final) * (i / num_episodes))
    epsilon = epsilon_final + (epsilon_initial - epsilon_final) * np.exp(-0.01 * i)


plt.plot(episode, eps_list)
plt.show()
