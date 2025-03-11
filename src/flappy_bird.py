from tkinter import Y
import pygame
import sys
import random

# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.08
FLAP_STRENGTH = -10
PIPE_SPEED = 5
PIPE_GAP = 150

# Colors
WHITE = (255, 255, 255)
SKY_BLUE = (135, 206, 235)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class Bird:
    def __init__(self):
        self.x = 50
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.acceleration = 0

    def flap(self):
        self.acceleration = 0
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.acceleration += GRAVITY
        self.velocity += self.acceleration
        self.y += self.velocity

class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.height = random.randint(100, 400)
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED

    def offscreen(self):
        return self.x < -50

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.bird = Bird()
        self.pipes = [Pipe()]
        self.score = 0

    def reset(self):
        self.bird = Bird()
        self.pipes = [Pipe()]
        self.score = 0
        return self.get_state()

    def get_state(self):
        # State: [bird_y, bird_velocity, distance_to_next_pipe, pipe_top, pipe_bottom]
        if len(self.pipes) > 0:
            next_pipe = self.pipes[0]
            horizontal_dist = next_pipe.x - self.bird.x
            vertical_dist_top = self.bird.y - (next_pipe.height - PIPE_GAP // 2)
            vertical_dist_bottom = (next_pipe.height + PIPE_GAP // 2) - self.bird.y
            return [self.bird.y, self.bird.velocity, horizontal_dist, vertical_dist_top, vertical_dist_bottom]
        else:
            return [0] * 5  # Default state if no pipes

    def step(self, action):
        # Action: 0 = do nothing, 1 = flap
        if action == 1:
            self.bird.flap()

        # Update bird and pipes
        self.bird.update()
        for pipe in self.pipes:
            pipe.update()

        # Check collisions
        if self.bird.y < 0:
            done = True
            reward = -1000
        else:
            done = False
            reward = 0.1  # Small reward for surviving

        # Add new pipes and remove offscreen pipes
        if self.pipes[-1].x < SCREEN_WIDTH - 200:
            self.pipes.append(Pipe())
        self.pipes = [pipe for pipe in self.pipes if not pipe.offscreen()]

        # Check scoring
        if not self.pipes[0].passed and self.pipes[0].x < self.bird.x:
            self.score += 1
            self.pipes[0].passed = True
            reward = 10  # Big reward for passing a pipe

        # Get next state
        next_state = self.get_state()
        return next_state, reward, done

    def render(self):
        self.screen.fill(SKY_BLUE)
        # Draw bird
        pygame.draw.circle(self.screen, YELLOW, (self.bird.x, int(self.bird.y)), 15)
        # Draw pipes
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, GREEN, (pipe.x, 0, 50, pipe.height - PIPE_GAP // 2))
            pygame.draw.rect(self.screen, GREEN, (pipe.x, pipe.height + PIPE_GAP // 2, 50, SCREEN_HEIGHT))
        pygame.display.update()
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