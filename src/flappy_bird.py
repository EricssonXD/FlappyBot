import sys
import pygame
import random
import numpy as np
from pygame.sprite import Sprite, Group
import os
from config import *  # noqa: F403

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
# font_path = "assets/fonts/flappy-font.ttf"  # This works only if you run the script from the root directory
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
        self.flapping = False

    def flap(self):
        self.flapping = True

    def update(self):
        self.acceleration += GRAVITY
        self.velocity += self.acceleration

        if self.flapping:
            self.acceleration = 0
            self.velocity = -FLAP_STRENGTH
            self.flapping = False

        self.velocity = min(18, self.velocity)
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
    def __init__(self, training_mode=False):
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
            next_pipe = sorted(self.pipes, key=lambda p: p.rect.x)  # Closest pipe
            s = np.array(
                [
                    self.bird.rect.centery / SCREEN_HEIGHT,  # Normalized
                    self.bird.velocity / 10,  # Scaled
                    (next_pipe[0].rect.centerx - self.bird.rect.right) / SCREEN_WIDTH,
                    (next_pipe[0].rect.bottom) / SCREEN_HEIGHT,
                    (next_pipe[1].rect.top) / SCREEN_HEIGHT,
                ],
                dtype=np.float32,
            )
            # print(s)
            return s
        else:
            return np.zeros(5, dtype=np.float32)

    def step(self, action):
        reward = SURVIVAL_REWARD  # Survival reward

        # Action: 0 = no flap, 1 = flap
        if action == 1:
            self.bird.flap()
            reward += FLAP_REWARD  # Reduces reward for flapping

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
        if gameover:
            reward += COLLISION_REWARD
        elif (
            not self.pipes.sprites()[0].passed
            and self.pipes.sprites()[0].rect.right < self.bird.rect.left
        ):
            self.pipes.sprites()[0].passed = True
            self.score += 1
            reward += PIPE_PASSED_REWARD  # Passed a pipe
            if self.score > 100:
                gameover = True  # Stop the game after 100 pipes
                reward += 1000

        # Render if not in training mode
        if not self.training_mode:
            self.render()
        # self.render()

        if not self.training_mode:
            # self.clock.tick(TRAINING_FPS)
            self.clock.tick(GAME_FPS)
        else:
            self.clock.tick(TRAINING_FPS)

        return self._get_state(), reward, gameover

    def render(self):

        # First draw the background
        self.screen.fill(SKY_BLUE)

        # Draw the pipes and the bird
        self.all_sprites.draw(self.screen)

        # Draw the score
        score_text = font.render(f"{self.score}", False, BLACK)
        score_text_rect = score_text.get_rect()
        score_text_rect.centerx = self.screen.get_rect().centerx
        score_text_rect.top = 10
        self.screen.blit(score_text, score_text_rect)

        # Update display
        pygame.display.flip()


def gameLoop(game):
    action = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1
    state, reward, done = game.step(action)
    if done:
        game.reset()


def run():
    game = Game()
    while True:
        # start_time = time.time()
        gameLoop(game)
        # end_time = time.time()
        # print(f"Step took: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    run()
    pygame.quit()
    sys.exit()
