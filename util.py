import numpy as np

def valid_path(grid, start, goal):
    # Define the four possible movements (up, down, left, right)
    movements = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Initialize the queue with the starting position
    queue = [start]

    # Initialize the visited set with the starting position
    visited = set()
    visited.add(start)

    # Initialize the path dictionary with the starting position
    path = {start: None}

    # Search for the goal position using Breadth-First Search
    while queue:
        curr_pos = queue.pop(0)
        if curr_pos == goal:
            return True

        # Try each possible movement from the current position
        for movement in movements:
            next_pos = tuple([curr_pos[i] + movement[i] for i in range(2)])

            # Check if the next position is a valid move and has not been visited before
            if (0 <= next_pos[0] < grid.shape[0]
                    and 0 <= next_pos[1] < grid.shape[1]
                    and grid[next_pos] != -1
                    and next_pos not in visited):

                # Add the next position to the queue, visited set, and path dictionary
                queue.append(next_pos)
                visited.add(next_pos)
                path[next_pos] = curr_pos

    # If the goal position is not found, return false
    return False

def epsilon_greedy(Q, state, epsilon, env):
    np.random.seed()
    if np.random.random() < epsilon:
        return env.action_space_sample()
    else:
        # if there are multiple actions with the same max value, choose one of them randomly
        randargmax = np.random.random(Q[state].shape) * (Q[state] == Q[state].max())
        return np.argmax(randargmax)

def optimal_policy(Q, state):
    return np.argmax(Q[state])