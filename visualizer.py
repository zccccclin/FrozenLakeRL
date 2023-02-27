import pygame

class FrozenLakeVisualizer:
    def __init__(self, env):
        self.env = env
        # self.colors = {
        #     0: (173, 216, 230),    # frozen lake
        #     -1: (255, 0, 0),       # holes
        #     1: (0, 255, 0),        # frisbee
        # }
        
        self.tile_size = 100
        self.screen_width = self.tile_size * self.env.map_size
        self.screen_height = self.tile_size * self.env.map_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Frozen Lake")
        self.loadImages()
        self.images = {
            2 : self.robot,
            1 : self.goal,
            0 : self.ice,
            -1 : self.hole,
        }
    
    def draw(self, map, robot_pos):
        self.screen.fill((0, 0, 0))  # fill the screen with black
        # for i in range(self.env.map_size):
        #     for j in range(self.env.map_size):
        #         state = map[i][j]
        #         color = self.colors[state]
        #         rect = pygame.Rect(j*self.tile_size, i*self.tile_size, self.tile_size, self.tile_size)
        #         pygame.draw.rect(self.screen, color, rect)
        i = robot_pos[0]
        j = robot_pos[1]
        for row in range(len(map)):
            for col in range(len(map[row])):
                tile = map[row][col]
                tile_rect = pygame.Rect(col*self.tile_size, row*self.tile_size, self.tile_size, self.tile_size)
                if row == i and col == j:
                    # Robot tile
                    self.screen.blit(self.images[2], tile_rect)
                elif tile == 1:
                    # Goal tile
                    self.screen.blit(self.images[1], tile_rect)
                elif tile == -1:
                    # Hole tile
                    self.screen.blit(self.images[-1], tile_rect)
                else:
                    # Frozen tile
                    self.screen.blit(self.images[0], tile_rect)
        
        # draw the player
        # player_pos = robot_pos
        # player_rect = pygame.Rect(player_pos[1]*self.tile_size, player_pos[0]*self.tile_size, self.tile_size, self.tile_size)
        # self.screen.blit(self.screen, self.images[2], player_rect)
        pygame.display.flip()  # update the screen
        self.delay(100)
    
    def close(self):
        pygame.quit()

    def delay(self, ms):
        pygame.time.delay(ms)

    def loadImages(self):
        goal = pygame.image.load("images/frisbee.png").convert()
        robot = pygame.image.load("images/robot.png").convert()
        ice = pygame.image.load("images/ice.png").convert()
        hole = pygame.image.load("images/hole.png").convert()
        self.goal = pygame.transform.scale(goal, (self.tile_size, self.tile_size))
        self.robot = pygame.transform.scale(robot, (self.tile_size, self.tile_size))
        self.ice = pygame.transform.scale(ice, (self.tile_size, self.tile_size))
        self.hole = pygame.transform.scale(hole, (self.tile_size, self.tile_size))