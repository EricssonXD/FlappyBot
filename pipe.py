import pygame
import random

class Pipe:
    def __init__(self, x, y, inverted=False):
        self.x = x
        self.y = y
        self.inverted = inverted
        self.pipe_surface = pygame.image.load('assets/pipe-green.png').convert()
        self.pipe_surface = pygame.transform.scale2x(self.pipe_surface)
        self.rect = self.pipe_surface.get_rect(midtop=(self.x, self.y))

    def draw(self, screen):
        if self.inverted:
            rotated_pipe = pygame.transform.flip(self.pipe_surface, False, True)
            screen.blit(rotated_pipe, self.rect)
        else:
            screen.blit(self.pipe_surface, self.rect)
