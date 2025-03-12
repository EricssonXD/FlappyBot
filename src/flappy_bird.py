import sys
import pygame
import random
import numpy as np
from pygame.sprite import Sprite, Group
import os

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.08
FLAP_STRENGTH = -9
PIPE_SPEED = 6
PIPE_GAP = 150
PIPE_SPACING = 200  # Distance between pipes

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
SKY_BLUE = (135, 206, 235)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

pygame.init()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(BASE_DIR, "..", "assets", "fonts", "flappy-font.ttf")
font = pygame.font.Font(font_path, 48)

# pygame.font.init()  # you have to call this at the start,
# if you want to use this module.
# font = pygame.font.SysFont("Comic Sans MS", 30)


class Bird(Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(self.image, YELLOW, (15, 15), 15)
        self.rect = self.image.get_rect(center=(50, SCREEN_HEIGHT // 2))
        self.velocity = 0
        self.acceleration = 0

    def flap(self):
        self.acceleration = 0
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.acceleration += GRAVITY
        self.velocity += self.acceleration
        self.rect.y += self.velocity


class Pipe(Sprite):
    def __init__(self, inverted=False):
        super().__init__()
        self.image = pygame.Surface((50, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.passed = False

        if inverted:
            # Top pipe
            self.rect.bottomleft = (
                SCREEN_WIDTH,
                random.randint(100, 400) - PIPE_GAP // 2,
            )
        else:
            # Bottom pipe
            self.rect.topleft = (SCREEN_WIDTH, random.randint(100, 400) + PIPE_GAP // 2)

    def update(self):
        self.rect.x -= PIPE_SPEED
        if self.rect.right < 0:
            self.kill()  # Remove offscreen pipes
            del self


class Game:
    def __init__(self, training_mode=True):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.bird = Bird()
        self.pipes = Group()
        self.all_sprites = Group()
        self.all_sprites.add(self.bird)
        self.score = 0
        self.training_mode = training_mode  # Disable rendering for faster training

    def reset(self):
        self.all_sprites.empty()
        self.pipes.empty()
        self.bird = Bird()
        self.all_sprites.add(self.bird)
        self._add_pipe()
        self.score = 0
        return self._get_state()

    def _add_pipe(self):
        # Add top and bottom pipes as a pair
        y = random.randint(100, 400)
        top_pipe = Pipe(inverted=True)
        top_pipe.rect.bottom = y - PIPE_GAP // 2
        bottom_pipe = Pipe()
        bottom_pipe.rect.top = y + PIPE_GAP // 2
        self.pipes.add(top_pipe, bottom_pipe)
        self.all_sprites.add(top_pipe, bottom_pipe)

    def _get_state(self):
        # State: [bird_y, bird_velocity, next_pipe_x, next_pipe_top, next_pipe_bottom]
        if len(self.pipes) > 0:
            next_pipe = sorted(self.pipes, key=lambda p: p.rect.x)[0]  # Closest pipe
            return np.array(
                [
                    self.bird.rect.centery / SCREEN_HEIGHT,  # Normalized
                    self.bird.velocity / 10,  # Scaled
                    (next_pipe.rect.centerx - self.bird.rect.centerx) / SCREEN_WIDTH,
                    (next_pipe.rect.bottom - self.bird.rect.centery) / SCREEN_HEIGHT,
                    (next_pipe.rect.top - self.bird.rect.centery) / SCREEN_HEIGHT,
                ],
                dtype=np.float32,
            )
        else:
            return np.zeros(5, dtype=np.float32)

    def step(self, action):
        # Action: 0 = no flap, 1 = flap
        if action == 1:
            self.bird.flap()

        # Update sprites
        self.all_sprites.update()

        # Add new pipes
        if (
            len(self.pipes) == 0
            or self.pipes.sprites()[-1].rect.x < SCREEN_WIDTH - PIPE_SPACING
        ):
            self._add_pipe()

        # Check collisions
        collision = pygame.sprite.spritecollideany(self.bird, self.pipes)
        out_of_bounds = self.bird.rect.top < 0 or self.bird.rect.bottom > SCREEN_HEIGHT
        gameover = collision or out_of_bounds

        # Calculate reward
        reward = 0.1  # Survival reward
        if gameover:
            reward = -1000
        elif (
            self.pipes.sprites()[0].rect.right < self.bird.rect.left
            and not self.pipes.sprites()[0].passed
        ):
            self.pipes.sprites()[0].passed = True
            self.score += 1
            reward = 10  # Passed a pipe

        # Render if not in training mode
        if not self.training_mode:
            self.render()

        return self._get_state(), reward, gameover

    def render(self):
        score_text = font.render(f"{self.score}", False, BLACK)
        score_text_rect = score_text.get_rect()
        score_text_rect.centerx = self.screen.get_rect().centerx
        score_text_rect.top = 10

        self.screen.fill(SKY_BLUE)
        self.all_sprites.draw(self.screen)
        self.screen.blit(score_text, score_text_rect)

        pygame.display.flip()
        self.clock.tick(30)


def run():
    game = Game()
    while True:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
        state, reward, done = game.step(action)
        game.render()
        if done:
            game.reset()


if __name__ == "__main__":
    run()
    pygame.quit()
    sys.exit()
