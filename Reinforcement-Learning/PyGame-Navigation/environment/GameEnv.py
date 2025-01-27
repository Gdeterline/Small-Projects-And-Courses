import pygame
from InteractiveObject import InteractiveObject

HEIGHT = 800
WIDTH = 1200

pygame.init()


clock = pygame.time.Clock()



class GameEnv():
    def __init__(self):
        pygame.init()
        self.obj = InteractiveObject()
        self.screen = pygame.display.set_mode((HEIGHT, WIDTH))


    def run_env(self):

        running = True
        while running:
            # Set FPS
            clock.tick(1)


            self.screen.fill((0, 0, 0))
            # Draw the player on the screen
            self.screen.blit(self.obj.surf, self.obj.rect)
            # Update the display
            pygame.display.flip()

            # Look at every event in the queue
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

run_game = GameEnv()
run_game.run_env()
pygame.quit()