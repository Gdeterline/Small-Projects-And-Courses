import pygame

class InteractiveObject():
    def __init__(self):
        super(InteractiveObject, self).__init__()
        self.surf = pygame.Surface((50, 50))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect(center=(500, 500))   # need to check where to store the position - which class
        