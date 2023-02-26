import Environment as fl
import numpy as np
import matplotlib.pyplot as plt

class MonteCarloNoES:
    ############################################################
    #                     Initialization                       #
    ############################################################
    def __init__(self, env, gamma, epsilon, num_episodes, num_steps):
        self.env = env
        self.rewards = 0
        self.avg_reward = []

        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.num_steps = num_steps

        # Initialize Q-table, Return table and visitation count table
        self.Q = np.zeros([env.num_obs, env.num_actions])
        self.Return = np.zeros([env.num_obs, env.num_actions])
        self.N = np.zeros([env.num_obs, env.num_actions])
        print("Agent initialized")
        
    # e-Greedy Policy function
    def epsilon_greedy(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space_sample()
        else:
            return np.argmax(self.Q[state])
        
    def run(self):
        for eps in range(self.num_episodes):
            # Initialize episode
            state = self.env.reset()
            done = False
            episode_rewards = []
            state_action_pairs = []

            # Loop over steps in the episode
            for step in range(self.num_steps):
                state = self.env.pos_to_state(state)
                # Choose action
                action = self.epsilon_greedy(state)
                # Take action
                next_state, reward, done = self.env.step(action)
                # Store episode reward and State-Action pair
                episode_rewards.append(reward)
                state_action_pairs.append((state, action))
                # Update number of times each state-action pair has been visited
                self.N[state][action] += 1

                # If the episode is finished
                if done:
                    G = 0
                    for t in range(len(state_action_pairs) - 1, -1, -1):
                        state_t, action_t = state_action_pairs[t]
                        G = self.gamma * G + episode_rewards[t]
                        # Check if state-action pair has been visited before
                        if (state_t, action_t) not in state_action_pairs[:t]:
                            self.Return[state_t][action_t] += G
                            self.Q[state_t][action_t] = self.Return[state_t][action_t]/self.N[state_t][action_t]
                    break

                # Update state
                state = next_state

            # calculate average reward
            self.rewards += sum(episode_rewards)
            self.avg_reward.append(self.rewards/(eps+1))

            if eps % 1000 == 0:
                print("Episode: ", eps, "Total reward: ", sum(episode_rewards))
            
        return self.Q
    
if __name__ == "__main__":
    # Initialize environment
    env = fl.FrozenLake(2, 10, False)
    env.reset()

    # Initialize Monte Carlo agent
    agent = MonteCarloNoES(env, 0.9, 0.1, 10000, 100)

    # Run agent
    Q = agent.run()

    # Print Q-table
    print(Q)

    # Plot average reward
    plt.plot(agent.avg_reward)
    plt.show()