import Environment as fl
import util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import argparse
import pandas as pd

class MonteCarlo_Agent:
    def __init__(self, env, gamma, epsilon, num_episodes, num_steps, visualize=False, testing=False):
        self.env = env
        self.visualize = visualize
        self.testing = testing
        self.test_num = 0
        self.num_episodes = num_episodes
        self.num_steps = num_steps

        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table, Return table and visitation count table
        self.Q = np.zeros((self.env.num_obs, self.env.num_actions))
        self.Return = np.zeros([self.env.num_obs, self.env.num_actions])
        self.N = np.zeros([self.env.num_obs, self.env.num_actions])

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
            # Initialize episode
            state = self.env.reset()
            done = False
            episode_rewards = []
            state_action_pairs = []

            # Loop over steps in the episode
            for step in range(self.num_steps):
                state = self.env.pos_to_state(state)
                # Choose action
                action = util.epsilon_greedy(self.Q , state, self.epsilon, self.env)
                # Take action
                next_state, reward, done = self.env.step(action)
                if self.visualize:
                    self.env.render()
                # Store episode reward and State-Action pair
                episode_rewards.append(reward)
                state_action_pairs.append((state, action))
                # Update number of times each state-action pair has been visited
                self.N[state][action] += 1

                # If the episode is finished
                if done or step == self.num_steps - 1:
                    if reward <= 0:
                        self.train_fail_cnt += 1
                    if reward == 1:
                        self.train_success_cnt += 1
                    self.step_cnt += [step + 1]
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
            self.episode_cnt += [eps+1]
            self.total_rewards += sum(episode_rewards)
            self.rewards += [self.total_rewards]
            avg_reward = self.total_rewards/(eps+1)
            self.avg_reward.append(avg_reward)
            if (eps + 1) % 1000 == 0:
                print("Episode: ", eps+1, "Avg reward: ", avg_reward)

        # Data logging
        train_succ_rate = self.train_success_cnt * 100 / self.num_episodes
        print("Train success: ", self.train_success_cnt, "Train fail: ", self.train_fail_cnt)
        print("Training success rate: ", train_succ_rate) 

        # Save data
        self.save("train") 

        return self.Q
    
    # Test function 
    def test(self, model_fname, test_num):
        self.test_num = test_num
        fname = os.path.join("Models", model_fname)
        self.Q = np.load(fname)
        self.visualize = True
        for eps in range(test_num):
            pos = self.env.reset()
            done = False
            episode_rewards = []
            for step in range(self.num_steps):
                state = self.env.pos_to_state(pos)
                action = util.optimal_policy(self.Q, state)
                self.test_route.append([pos.tolist(), action+1])
                next_state, reward, done = self.env.step(action)
                if self.visualize:
                    self.env.render()
                episode_rewards.append(reward)
                if done:
                    if reward == -1:
                        self.test_fail_cnt += 1
                    if reward == 1:
                        self.test_success_cnt += 1
                    self.step_cnt += [step+1]
                    if eps == 0:
                        self.save("test")
                    self.test_route = []
                    break
                pos = next_state
            if not done:
                self.test_fail_cnt += 1
                self.step_cnt += [self.num_steps]
                self.test_route = []
            
            # calculate average reward
            self.episode_cnt += [eps+1]
            self.total_rewards += sum(episode_rewards)
            self.rewards += [self.total_rewards]
            avg_reward = self.total_rewards/(eps+1)
            self.avg_reward.append(avg_reward)

        # Data logging
        print("Average reward: ", avg_reward)
        test_succ = self.test_success_cnt * 100 / test_num
        print("Test success: ", self.test_success_cnt, "Test fail: ", self.test_fail_cnt)
        print("Test success rate: ", test_succ) 

        
    def save(self, train_test="train"):
        log_folder = "Logging"
        model_folder = "Models"
        if not os.path.exists(log_folder):
            os.makedirs("Logging")
        if not os.path.exists(model_folder):
            os.makedirs("Models")
        

        if train_test == "train":
            # Save Q-table and image of optimal policy
            trained_Q_fname = os.path.join(model_folder, "T{}_MC.npy".format(self.env.task))
            policyImage_fname = os.path.join(log_folder, "T{}_MC_Policy.png".format(self.env.task))
            np.save(trained_Q_fname, self.Q)
            self.env.savePolicyImage(self.Q, policyImage_fname) 
            # Save training data
            name = ["Episode", "Avg Reward", "Steps", "Total Reward"]
            dictionary = {"Episode": self.episode_cnt, "Avg Reward": self.avg_reward, "Steps": self.step_cnt, "Total Reward": self.rewards}
            dataframe = pd.DataFrame(dictionary)
            dataframe.to_csv(os.path.join(log_folder, "T{}_MC_Log.csv".format(self.env.task)), index=False, header=True)
        elif train_test == "test":
            # Save route taken by agent
            routeImage_fname = os.path.join(log_folder, "T{}_MC_Route.png".format(self.env.task))
            self.env.saveRouteImage(self.test_route, routeImage_fname)
    
    def plot_results(self):
        fig, axs = plt.subplots(2,2)
        fig.suptitle("Task {} Monte Carlo {}".format(self.env.task, "Training" if not self.testing else "Test"))
        # Plot average reward vs episode
        axs[0,0].plot(self.avg_reward)
        axs[0,0].set_title("Average reward vs Episode".format(self.env.task))
        axs[0,0].set(xlabel="Episode", ylabel="Average reward")
        axs[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # Plot success failure bar graph
        if not self.testing:
            axs[0,1].set_title("Training Success and Fail %")
            axs[0,1].bar(["Success", "Fail"], [self.train_success_cnt/self.num_episodes, self.train_fail_cnt/self.num_episodes], color=["green", "red"])
        else:
            axs[0,1].set_title("Test Success and Fail %")
            axs[0,1].bar(["Success", "Fail"], [self.test_success_cnt/self.test_num, self.test_fail_cnt/self.test_num], color=["green", "red"])
        axs[0,1].set(ylabel="Percentage")
        # Plot steps vs episode
        axs[1,0].plot(self.step_cnt)
        axs[1,0].set_title("Steps vs Episode")
        axs[1,0].set(xlabel="Episode", ylabel="Steps")
        axs[1,0].yaxis.set_major_formatter(FormatStrFormatter('%d'))
        # Plot cummulative reward vs episode
        axs[1,1].plot(self.rewards)
        axs[1,1].set_title("Cummulative reward vs Episode")
        axs[1,1].set(xlabel="Episode", ylabel="Cummulative reward")
        plt.tight_layout(pad=0.5, w_pad=1, h_pad=1.0)
        plt.show()
        if not self.testing:
            fig.savefig("Logging/T{}_MC_Train_Plot.png".format(self.env.task))
        else:
            fig.savefig("Logging/T{}_MC_Test_Plot.png".format(self.env.task))
        plt.waitforbuttonpress(0)
        plt.close()

if __name__ == "__main__":
    # Define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--map_size", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--test_num", type=int, default=10)
    parser.add_argument("--model", type=str, default="T1_MC.npy")
    args = parser.parse_args()

    # Initialize environment
    env = fl.FrozenLake(task_num=args.task, map_size=args.map_size)
    env.reset()

    # Initialize Monte Carlo agent
    agent = MonteCarlo_Agent(env, args.gamma, args.epsilon, args.num_episodes, args.num_steps, visualize=args.visualize, testing=args.test)

    # Run agent
    if not agent.testing:
        Q = agent.train()
    else:
        agent.test(args.model, args.test_num)

    # Print Q-table
    #print(Q)

    # Plot average reward
    agent.plot_results()

