import environment as fl
import numpy as np
import util
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import argparse
import pandas as pd

class SARSA_Agent:
    def __init__(self, env, lr, gamma, epsilon, decay, num_episodes, num_steps, visualize=False, testing=False):
        self.env = env
        self.visualize = visualize
        self.testing = testing
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.decay = decay

        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.epsilon_initial = epsilon

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
        self.DNF = 0
        self.DNF_percent = []
        self.MSE_diff = []
        self.first_goal_reached = False
        print("Agent initialized")

    # Main training function
    def train(self):
        pos_reward = 0
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
                    
                # Visualize environment
                if self.visualize:
                    self.env.render()

                # Break if episode is done
                if done:
                    if episode_reward <= 0:
                        self.train_fail_cnt += 1
                    else:
                        pos_reward += 1
                        if self.decay:
                            # linear decay
                            #self.epsilon = self.epsilon_initial - self.epsilon_initial * ((pos_reward)/5000)
                            # exponential decay
                            self.epsilon = self.epsilon_initial * np.exp(-0.001 * (pos_reward))
                            print(self.epsilon)

                        self.train_success_cnt += 1
                        if not self.first_goal_reached:
                            print("First goal reached at episode: ", eps+1)
                            self.first_goal_reached = True
                    self.step_cnt += [step+1]
                    break
                if step == self.num_steps-1:
                    self.DNF += 1
                    self.train_fail_cnt += 1
                    self.step_cnt += [step+1]
                    break
                # Update state and action
                state = next_state
                action = next_action

            
            # calculate log data
            mse = np.sum(np.power(self.Q,2)) / (self.env.map.shape[0] * self.env.map.shape[1])
            self.MSE_diff += [mse]
            self.episode_cnt += [eps+1]
            self.DNF_percent += [self.DNF*100/(eps+1)]
            self.total_rewards += episode_reward
            self.rewards.append(self.total_rewards)
            self.avg_reward.append(self.total_rewards/(eps+1))
            if eps % 1000 == 0:
                print("Episode: ", eps+1, "Average reward: ", self.avg_reward[-1], "DNF percent: ", self.DNF_percent[-1])
                    # Update epsilon


        # Data logging
        train_succ_rate = self.train_success_cnt * 100 / self.num_episodes
        print("Train success: ", self.train_success_cnt, "Train fail: ", self.train_fail_cnt)
        print("Training success rate: ", train_succ_rate)

        # Save data
        self.save("train")
        return self.Q

    def test(self, model_fname, test_num, visualize):
        self.test_num = test_num
        fname = os.path.join("Models", model_fname)
        self.Q = np.load(fname)
        self.visualize = visualize
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
                if done or step == self.num_steps - 1:
                    if reward <= 0:
                        self.test_fail_cnt += 1
                    if reward > 0:
                        self.test_success_cnt += 1
                    self.step_cnt += [step+1]
                    if eps == 0:
                        self.save("test")
                    self.test_route = []
                    break
                pos = next_state
            
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
        log_folder = "Logging/SARSA"
        model_folder = "Models"
        if not os.path.exists(log_folder):
            os.makedirs("Logging/SARSA")
        if not os.path.exists(model_folder):
            os.makedirs("Models")
        

        if train_test == "train":
            # Save Q-table and image of optimal policy
            trained_Q_fname = os.path.join(model_folder, "T{}_SARSA.npy".format(self.env.task))
            policyImage_fname = os.path.join(log_folder, "T{}_SARSA_Policy.png".format(self.env.task))
            np.save(trained_Q_fname, self.Q)
            self.env.savePolicyImage(self.Q, policyImage_fname) 
            # Save training data
            name = ["Episode", "Avg Reward", "Steps", "Total Reward", "Q-MSE"]
            dictionary = {"Episode": self.episode_cnt, "Avg Reward": self.avg_reward, "Steps": self.step_cnt, "Total Reward": self.rewards, "Q-MSE": self.MSE_diff}
            dataframe = pd.DataFrame(dictionary)
            dataframe.to_csv(os.path.join(log_folder, "T{}_SARSA_Log.csv".format(self.env.task)), index=False, header=True)
        elif train_test == "test":
            # Save route taken by agent
            routeImage_fname = os.path.join(log_folder, "T{}_SARSA_Route.png".format(self.env.task))
            self.env.saveRouteImage(self.test_route, routeImage_fname)
    
    def plot_results(self):
        fig, axs = plt.subplots(2,2)
        fig.suptitle("Task {} SARSA {}".format(self.env.task, "Training" if not self.testing else "Test"))
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
        axs[1,0].scatter(self.episode_cnt,self.step_cnt, marker=".")
        axs[1,0].set_title("Steps vs Episode")
        axs[1,0].set(xlabel="Episode", ylabel="Steps")
        axs[1,0].yaxis.set_major_formatter(FormatStrFormatter('%d'))
        # Plot cummulative reward vs episode
        axs[1,1].plot(self.DNF_percent)
        axs[1,1].set_title("DNF percent vs Episode")
        axs[1,1].set(xlabel="Episode", ylabel="DNF Percent")
        plt.tight_layout(pad=0.5, w_pad=1, h_pad=1.0)
        plt.show()
        if not self.testing:
            fig.savefig("Logging/SARSA/T{}_SARSA_Train_Plot.png".format(self.env.task))
        else:
            fig.savefig("Logging/SARSA/T{}_SARSA_Test_Plot.png".format(self.env.task))
        plt.waitforbuttonpress(0)
        plt.close()


if __name__ == "__main__":
    # Define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--map_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--decay", type=bool, default=False)
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--model", type=str, default="T1_SARSA.npy")
    args = parser.parse_args()

    # Initialize environment
    env = fl.FrozenLake(task_num=args.task, map_size=args.map_size)
    env.reset()

    # Initialize Monte Carlo agent
    agent = SARSA_Agent(env, args.lr, args.gamma, args.epsilon, args.decay, args.num_episodes, args.num_steps, visualize=args.visualize, testing=args.test)

    # Run agent
    if not agent.testing:
        Q = agent.train()
    else:
        print(args.visualize)
        agent.test(args.model, args.test_num, args.visualize)

    # Print Q-table
    #print(Q)

    # Plot average reward
    agent.plot_results()