import pygame

class FrozenLakeVisualizer:
    def __init__(self, env):
        self.env = env
        self.colors = {
            0: (173, 216, 230),    # frozen lake
            -1: (255, 0, 0),       # holes
            1: (0, 255, 0),        # frisbee
        }
        self.tile_size = 100
        self.screen_width = self.tile_size * self.env.size
        self.screen_height = self.tile_size * self.env.size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Frozen Lake")
    
    def draw(self, map, robot_pos):
        self.screen.fill((0, 0, 0))  # fill the screen with black
        for i in range(self.env.size):
            for j in range(self.env.size):
                state = map[i][j]
                color = self.colors[state]
                rect = pygame.Rect(j*self.tile_size, i*self.tile_size, self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, color, rect)
        
        # draw the player
        player_pos = robot_pos
        player_rect = pygame.Rect(player_pos[1]*self.tile_size, player_pos[0]*self.tile_size, self.tile_size, self.tile_size)
        pygame.draw.rect(self.screen, (255, 255, 255), player_rect)
        pygame.display.flip()  # update the screen
        self.delay(100)
    
    def close(self):
        pygame.quit()

    def delay(self, ms):
        pygame.time.delay(ms)