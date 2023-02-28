import pygame

class FrozenLakeVisualizer:
    def __init__(self, map_size):
        self.tile_size = 100
        self.screen_width = self.tile_size * map_size
        self.screen_height = self.tile_size * map_size
        pygame.init()
        pygame.display.set_caption("Frozen Lake")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.loadImages()
        self.images = {
            2 : self.robot,
            1 : self.goal,
            0 : self.ice,
            -1 : self.hole,
            "up" :self.up,
            "down" :self.down,
            "left" :self.left,
            "right" :self.right,
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

        pygame.display.flip()  # update the screen
        self.delay(100)
    
    def saveImage(self, map, filename):
        self.screen.fill((0, 0, 0))  # fill the screen with black
        for row in range(len(map)):
            for col in range(len(map[row])):
                tile = map[row][col]
                tile_rect = pygame.Rect(col*self.tile_size, row*self.tile_size, self.tile_size, self.tile_size)
                self.screen.blit(self.images[tile], tile_rect)
        pygame.display.flip()  # update the screen
        pygame.image.save(self.screen, filename)

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

        up = pygame.image.load("images/up.png").convert()
        down = pygame.image.load("images/down.png").convert()
        left = pygame.image.load("images/left.png").convert()
        right = pygame.image.load("images/right.png").convert()
        self.up = pygame.transform.scale(up, (self.tile_size, self.tile_size))
        self.down = pygame.transform.scale(down, (self.tile_size, self.tile_size))
        self.left = pygame.transform.scale(left, (self.tile_size, self.tile_size))
        self.right = pygame.transform.scale(right, (self.tile_size, self.tile_size))
    
