import environment as fl

i = int(input("Please input task num: "))
while not (i == 1 or i == 2):
    i = int(input("Wrong input, please enter '1' or '2': "))
if i == 1:
    env = fl.FrozenLake(task_num=1, map_size=4)
elif i == 2:
    env = fl.FrozenLake(task_num=2, map_size=10)
robot_pos = env.robot_pos
env.map[robot_pos[0],robot_pos[1]] = 2
if i == 1:
    env.visualizer.saveImage(env.map,"images/4x4.jpg")
elif i == 2:
    env.visualizer.saveImage(env.map,"images/10x10.jpg")