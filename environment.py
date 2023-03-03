import numpy as np 
import visualizer
import util
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

class FrozenLake:
    ############################################################
    #                     Initialization                       #
    ############################################################
    def __init__(self, task_num=1, map_size=4):
        self.task = task_num
        self.map_size = map_size
        self.start_pos = (0,0)
        self.goal_pos = (self.map_size-1,self.map_size-1)

        # Initialization
        self.map = self.generate_map()
        self.action_space = {0:[-1,0], # up
                             1:[1,0], # down
                             2:[0,-1], # left
                             3:[0,1]} # right
        
        self.num_actions = len(self.action_space)
        self.num_obs = self.map.size
        self.robot_pos = np.array(self.start_pos)        
        self.visualizer = visualizer.FrozenLakeVisualizer(self.map_size)
        print("Environment initialized")

    ############################################################
    #                           Actions                        #
    ############################################################
    def action_space_sample(self):
        np.random.seed()
        return np.random.randint(0,4)

    def pos_to_state(self, pos):
        return pos[0]*self.map_size + pos[1]

    def reset(self):
        self.robot_pos = np.array(self.start_pos)
        return self.robot_pos

    def step(self, action):
        # Move robot
        action = self.action_space[action]
        self.robot_pos += action

        # Check if robot is out of bounds
        if self.robot_pos[0] < 0 or self.robot_pos[0] >= self.map_size or self.robot_pos[1] < 0 or self.robot_pos[1] >= self.map_size:
            self.robot_pos -= action
            return self.robot_pos, 0, False

        # Check if robot is in hole
        if self.map[self.robot_pos[0]][self.robot_pos[1]] == -1:
            return self.robot_pos, -1, True

        # Check if robot is at goal
        if self.map[self.robot_pos[0]][self.robot_pos[1]] == 1:
            return self.robot_pos, 1, True

        return self.robot_pos, 0, False
    
    def render(self):
        self.visualizer.draw(self.map, self.robot_pos)  


    ############################################################
    #                    Helper Functions                      #
    ############################################################
    def savePolicyImage(self, Q, filename):
        optimalPolicyMap = self.map.tolist()
        actions = {1:"up", 2:"down", 3:"left", 4:"right"}
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.map[i][j] == 0:
                    state = Q[self.pos_to_state((i,j))]
                    if np.all(state == 0):
                        optimalPolicyMap[i][j] = 0
                    else:
                        optimalPolicyMap[i][j] = actions[np.argmax(state)+1]
        self.visualizer.saveImage(optimalPolicyMap, filename)

    def saveRouteImage(self, route, filename):
        routeMap = self.map.tolist()
        actions = {1:"up", 2:"down", 3:"left", 4:"right"}
        for i in route:
            routeMap[i[0][0]][i[0][1]] = actions[i[1]]
        self.visualizer.saveImage(routeMap, filename)

    # Generate map based on task 1 or task 2
    def generate_map(self):
        # Task 1 with fixed hole location
        if self.task == 1:
            return np.array([
                [0,0,0,0],
                [0,-1,0,-1],
                [0,0,0,-1],
                [-1,0,0,1]])

        # Task 2 with randomly generated hole location
        elif self.task == 2:
            map = np.zeros((self.map_size,self.map_size))
            map[-1][-1] = 1
            holes = self.map_size*self.map_size//4
            # set seed for reproducibility
            np.random.seed(5) 
            
            # add holes to map if its valid
            while holes > 0:
                x = np.random.randint(0,self.map_size)
                y = np.random.randint(0,self.map_size)
                if map[x][y] == 0 and (x, y) != self.start_pos:
                    map[x][y] = -1
                    holes -= 1
                    # Check if path is valid with BFS
                    if not util.valid_path(map, self.start_pos, self.goal_pos):
                        map[x][y] = 0
                        holes += 1
            return map
