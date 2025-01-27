import pygame
import math

class InteractiveObject():
    def __init__(self, x, y):
        super(InteractiveObject, self).__init__()
        # Car physics
        self.x = x
        self.y = y
        self.velocity = 0
        self.angle = 0
        self.max_velocity = 4.9

        # Car display
        self.surf = pygame.Surface((50, 50))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect(center=(x, y))   # need to check where to store the position - which class
        
    def turn_left(self):
        # If the car is moving forward, the angle increases by 5 degrees
        if self.velocity >= 0:
            self.angle += 2  
            # If the car is moving backward, the angle decreases by 5 degrees
        else:
            self.angle -= 2
    
    def turn_right(self):
        # Same here, but in the opposite direction
        if self.velocity >= 0:
            self.angle -= 2
        else:
            self.angle += 2
        
    def accelerate(self):
        if self.velocity <= self.max_velocity:
            self.velocity += 0.1
        
    def decelerate(self):
        if self.velocity >= 0.1:
            self.velocity -= 0.1  

    def move(self):
        self.x += self.velocity * math.cos(math.radians(self.angle))
        self.y -= self.velocity * math.sin(math.radians(self.angle))   

    def update(self):
        self.move()
        self.rect = self.surf.get_rect(center=(self.x, self.y))