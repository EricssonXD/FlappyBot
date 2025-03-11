import pygame

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bird_movement = 0
        self.gravity = 0.25
        self.bird_flap = -12
        self.rect = pygame.Rect(self.x, self.y, 34, 24)  # Assuming bird size

    def flap(self):
        self.bird_movement = self.bird_flap

    def update(self):
        self.bird_movement += self.gravity
        self.y += self.bird_movement
        if self.y < 0:
            self.y = 0
        self.rect.center = (self.x, self.y)
