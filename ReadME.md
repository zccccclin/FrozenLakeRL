# Frozen Lake Project

This is a project to explore 3 reinforcemnet learning techniques applied to the frozen lake problem.
1. First visit Monte Carlo without Exploring Start
2. SARSA
3. Q-Learning


# Problem

The problem involves a robot start at index (0,0) finding its way to goal index ((3,3) in task 1 and (9,9) in task 2) without falling into any holes. 
![4x4 map] (images/4x4.jpg)
![10x10 map] (images/10x10.jpg)

# Code structure
*environment.py* contains the custom environment built.
*util.py* contains extra functions required for running of agents and environment
*visualizer.py* contains custom built pygame visualizer for the environment
*MC_Agent.py* contains agent for first visit monte carlo without exploring start
*SARSA_Agent.py* contains agent for SARSA
*QL_Agent.py* contains agent for Q-Learning
*Logging* folder contains all the most recent optimal policy logs I have produced
*Models* folder contains the optimal policy models
# Running the codes

1. Monte Carlo without Exploring Start
For 4x4 
`python MC_Agent.py`
For 10x10
`python MC_Agent.py --task=2 --map_size=10`
2. SARSA
For 4x4 
`python SARSA_Agent.py`
For 10x10
`python SARSA_Agent.py --task=2 --map_size=10`
3. Q-Learning
For 4x4 
`python QL_Agent.py`
For 10x10
`python QL_Agent.py --task=2 --map_size=10`

# Extra parameters
Note the extra parameters that can be played with:
--gamma (sets the gamma value)
--epsilon (sets the epsilon)
--lr (sets the learning rate, only for SARSA and Q-Learning)
--decay (enable epsilon decay, must set --epsilon to initial epsilon)
--num_episodes (number of episodes)
--num_steps (maximum number of steps per episode)
--visualize (turn off or on visualization)
--test (to enable test mode, need to set --test_num and --model as well)
--test_num (number of test cases)
--model (model file name, place file in models folder first)