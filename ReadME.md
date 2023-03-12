
# Frozen Lake Project

This is a project to explore 3 reinforcemnet learning techniques applied to the frozen lake problem.
1. First visit Monte Carlo without Exploring Start
2. SARSA
3. Q-Learning


# Problem

The frozen lake problem involves an agent starting at index (0,0), with the objective of navigating to goal index without falling into any holes. The actions are limited to up, down, left and right.
There are two tasks in this project.
- Task 1 has its goal at index (3,3)

![4x4 map](https://github.com/zccccclin/FrozenLakeRL/blob/main/images/4x4.jpg)
- Task 2 has its goal at index (9,9)

![10x10 map](https://github.com/zccccclin/FrozenLakeRL/blob/main/images/10x10.jpg)

# Code structure
- **environment.py** contains the custom environment built.
- **util.py** contains extra functions required for running of agents and environment
- **visualizer.py** contains custom built pygame visualizer for the environment
- **MC_Agent.py** contains agent for first visit monte carlo without exploring start
- **SARSA_Agent.py** contains agent for SARSA
- **QL_Agent.py** contains agent for Q-Learning
- **Logging** folder contains all the most recent optimal policy and logs I have produced
- **Models** folder contains the optimal policy models
# Running the codes
### Packages installation ###
The project runs on python 3.6 and above and depend on following packages: 
- [ ] matplotlib
- [ ] pygame
- [ ] numpy
- [ ] pandas

Use `pip3 install ` to install these packages. 
#### Running Monte Carlo without Exploring Start Agent
For 4x4: `python MC_Agent.py`
For 10x10 :`python MC_Agent.py --task=2 --map_size=10`
#### Running SARSA Agent
For 4x4: `python SARSA_Agent.py`
For 10x10: `python SARSA_Agent.py --task=2 --map_size=10`
#### Running Q-Learning Agent
For 4x4: `python QL_Agent.py`
For 10x10: `python QL_Agent.py --task=2 --map_size=10`

**The above command lines runs the training function for the three agents with default parameters. To explore other feature please run with extra parameters. Parameter list can be found in the next section.
# Extra parameters
##### Note the extra parameters that can be played with:
1. `--gamma` (sets the gamma value)
2. `--epsilon` (sets the epsilon)
3. `--lr` (sets the learning rate, only for SARSA and Q-Learning)
4. `--decay` (enable epsilon decay, must set --epsilon to initial epsilon)
5. `--num_episodes` (number of episodes)
6. `--num_steps` (maximum number of steps per episode)
7. `--visualize` (turn off or on visualization)
8. `--test` (to enable test mode, need to set --test_num and --model as well)
9. `--test_num` (number of test cases)
10. `--model` (model file name, place file in models folder first)
