import Environment as fl
import numpy as np
import matplotlib.pyplot as plt

class MonteCarlo_Agent:
    def __init__(self, env, gamma, epsilon, num_episodes, num_steps):
        self.env = env
        self.rewards = 0
        self.avg_reward = []
        self.train_success = []
        self.test_success = []

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
    
    # Main function
    def train(self):
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
                self.env.render()
                # Store episode reward and State-Action pair
                episode_rewards.append(reward)
                state_action_pairs.append((state, action))
                # Update number of times each state-action pair has been visited
                self.N[state][action] += 1

                # If the episode is finished
                if done:
                    if reward == -1:
                        self.train_success += [0]
                    if reward == 1:
                        self.train_success += [1]
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
            
            # failed to reach frisbee within 100 steps
            if not done:
                self.train_success += [0]

            # calculate average reward
            self.rewards += sum(episode_rewards)
            avg_reward = self.rewards/(eps+1)
            self.avg_reward.append(avg_reward)
            if eps % 1000 == 0:
                print("Episode: ", eps, "Avg reward: ", avg_reward)

        # Data logging
        train_succ = sum(self.train_success) / len(self.train_success)
        print("Train success: ", self.train_success.count(1), "Train fail: ", self.train_success.count(0))
        print("Training success rate: ", train_succ*100)    
        return self.Q
    
    def test(self, model):
        self.Q = model

    
if __name__ == "__main__":
    # Initialize environment
    env = fl.FrozenLake(task_num=1, map_size=4, render=True)
    env.reset()

    # Initialize Monte Carlo agent
    agent = MonteCarlo_Agent(env, 0.9, 0.1, 10000, 100)

    # Run agent
    Q = agent.train()

    # Print Q-table
    #print(Q)

    # Plot average reward
    plt.plot(agent.avg_reward)
    plt.show()