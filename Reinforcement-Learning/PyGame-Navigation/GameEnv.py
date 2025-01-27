import pygame
from InteractiveObject import InteractiveObject

HEIGHT = 800
WIDTH = 1200

pygame.init()

screen = pygame.display.set_mode((HEIGHT, WIDTH))
clock = pygame.time.Clock()



class GameEnv():
    def __init__(self, obj):
        pygame.init()
        self.obj = InteractiveObject()

    clock.tick(1)

    running = True
    while running:
    # Look at every event in the queue
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

    