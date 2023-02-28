import Environment as fl
import numpy as np
import util

class SARSA_Agent:
    def __init__(self, env, lr, gamma, epsilon, num_episodes, num_steps, visualize=False, testing=False):
        self.env = env
        self.visualize = visualize
        self.testing = testing
        self.num_episodes = num_episodes
        self.num_steps = num_steps

        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr

        # Initialize Q-table
        self.Q = np.zeros([self.env.num_obs, self.env.num_actions])

        # Logging variables
        self.episode_cnt = []
        self.total_rewards = 0
        self.rewards = []
        self.avg_reward = []
        self.step_cnt = []
        self.train_success_cnt = 0
        self.train_fail_cnt = 0
        self.test_success_cnt = 0
        self.test_fail_cnt = 0
        self.test_route = []
        print("Agent initialized")

    # Main training function
    def train(self):
        for eps in range(self.num_episodes):
            # Reset environment
            done = False
            episode_reward = 0
            pos = self.env.reset()
            state = self.env.pos_to_state(pos)
            # Choose action from state using epsilon-greedy policy
            action = util.epsilon_greedy(self.Q, state, self.epsilon, self.env)
            
            for step in range(self.num_steps):
                # Take action and observe reward and next state
                next_pos, reward, done = self.env.step(action)
                next_state = self.env.pos_to_state(next_pos)
                episode_reward += reward
                # Choose next action from next state using epsilon-greedy policy
                next_action = util.epsilon_greedy(self.Q, next_state, self.epsilon, self.env)
                # Update Q-table
                self.Q[state, action] = self.Q[state, action] + self.lr * (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])
                # Update state and action
                state = next_state
                action = next_action
                
                # Visualize environment
                if self.visualize:
                    self.env.render()

                # Break if episode is done
                if done:
                    if episode_reward == 1:
                        self.train_success_cnt += 1
                    else:
                        self.train_fail_cnt += 1
                    self.step_cnt += [step+1]
                    break
            
            # failed to reach frisbee within 100 steps
            if not done:
                self.train_fail_cnt += 1
                self.step_cnt.append(self.num_steps)

            # calculate average reward
            self.total_rewards += episode_reward
            self.episode_cnt.append(eps+1)
            self.rewards.append(self.total_rewards)
            self.avg_reward.append(self.total_rewards/(eps+1))
            if eps % 1000 == 0:
                print("Episode: ", eps+1, "Average reward: ", self.avg_reward[-1])
            
        # Data logging
        train_succ_rate = self.train_success_cnt * 100 / self.num_episodes
        print("Train success: ", self.train_success_cnt, "Train fail: ", self.train_fail_cnt)
        print("Training success rate: ", train_succ_rate)

        return self.Q


if __name__ == "__main__":
    # Initialize environment
    env = fl.FrozenLake(task_num=2, map_size=10)
    env.reset()
    # Initialize agent
    agent = SARSA_Agent(env, lr=0.01, gamma=0.9, epsilon=0.9, num_episodes=100000, num_steps=1000, visualize=False)
    # Train agent
    Q = agent.train()
    #print(Q)