import numpy as np 
import visualizer
import util

class FrozenLake:
    def __init__(self, task=1, size=4, render=True):
        self.task = task
        self.size = size
        self.visualize = render
        self.start_pos = (0,0)
        self.goal_pos = (size-1,size-1)

        # Initialization
        self.map = self.generate_map()
        self.action_space = np.array([[-1,0], # up
                                      [1,0], # down
                                      [0,-1], # left
                                      [0,1]]) # right
        self.robot_pos = np.array(self.start_pos)        
        if self.visualize:
            self.visualizer = visualizer.FrozenLakeVisualizer(self)
            self.render()
        print("Environment initialized")
    

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
            map = np.zeros((self.size,self.size))
            map[-1][-1] = 1
            holes = self.size*self.size//4
            # set seed for reproducibility
            np.random.seed(999) 
            
            # add holes to map if its valid
            while holes > 0:
                x = np.random.randint(0,self.size)
                y = np.random.randint(0,self.size)
                if map[x][y] == 0 and (x, y) != self.start_pos:
                    map[x][y] = -1
                    holes -= 1
                    # Check if path is valid with BFS
                    if not util.valid_path(map, self.start_pos, self.goal_pos):
                        map[x][y] = 0
                        holes += 1
            return map

    def action_space_sample(self):
        np.random.seed()
        action = self.action_space[np.random.randint(0,4)]
        return action

    def reset(self):
        self.robot_pos = np.array(self.start_pos)
        return self.robot_pos

    def render(self):
        if self.visualize:
            self.visualizer.draw(self.map, self.robot_pos)

    def step(self, action):
        # Move robot
        self.robot_pos += action

        # Check if robot is out of bounds
        if self.robot_pos[0] < 0 or self.robot_pos[0] >= self.size or self.robot_pos[1] < 0 or self.robot_pos[1] >= self.size:
            self.robot_pos -= action
            return self.robot_pos, 0, False

        # Check if robot is in hole
        if self.map[self.robot_pos[0]][self.robot_pos[1]] == -1:
            return self.robot_pos, -1, True

        # Check if robot is at goal
        if self.map[self.robot_pos[0]][self.robot_pos[1]] == 1:
            return self.robot_pos, 1, True

        return self.robot_pos, 0, False
                