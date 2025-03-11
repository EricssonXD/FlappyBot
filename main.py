import pygame
import sys
import random
from bird import Bird
from pipe import Pipe

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
FPS = 60
GRAVITY = 0.25
BIRD_MOVEMENT = 0
PIPE_FREQUENCY = 1500  # milliseconds
last_pipe_time = pygame.time.get_ticks() - PIPE_FREQUENCY
SCORE = 0
HIGH_SCORE = 0

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Load images
bg_surface = pygame.image.load('assets/background-day.png').convert()
floor_surface = pygame.image.load('assets/base.png').convert()
floor_x_pos = 0
game_over_surface = pygame.image.load('assets/gameover.png').convert_alpha()

# Create game objects
bird = Bird(SCREEN_WIDTH // 5, SCREEN_HEIGHT // 2)
pipe_list = []

# Create a timer for the pipes
pipe_timer = pygame.USEREVENT
pygame.time.set_timer(pipe_timer, PIPE_FREQUENCY)

# Font
font = pygame.font.Font('assets/04B_19.TTF', 40)

def create_pipe():
    random_pipe_pos = random.choice(pipe_heights)
    bottom_pipe = Pipe(SCREEN_WIDTH, random_pipe_pos)
    top_pipe = Pipe(SCREEN_WIDTH, random_pipe_pos - 300, True)
    return bottom_pipe, top_pipe

def move_pipes(pipes):
    for pipe in pipes:
        pipe.x -= 5
    return pipes

def draw_pipes(pipes):
    for pipe in pipes:
        pipe.draw(screen)

def check_collision(pipes):
    global game_active
    for pipe in pipes:
        if bird.rect.colliderect(pipe.rect):
            game_active = False
            return
    if bird.rect.top <= -100 or bird.rect.bottom >= 450:
        game_active = False
        return

def rotate_bird(bird_surface):
    """Rotates the bird based on its movement."""
    new_bird = pygame.transform.rotozoom(bird_surface, -bird.bird_movement * 3, 1)
    return new_bird

def bird_animation():
    """Animates the bird."""
    new_bird_surface = bird_frames[bird_index]
    new_bird_rect = new_bird_surface.get_rect(center=(bird.x, bird.y))
    return new_bird_surface, new_bird_rect

def score_display(game_state):
    """Displays the score or game over message."""
    global SCORE, HIGH_SCORE
    if game_state == 'main_game':
        score_surface = font.render(str(int(SCORE)), True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(score_surface, score_rect)
    if game_state == 'game_over':
        score_surface = font.render(f'Score: {int(SCORE)}', True, (255, 255, 255))
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(score_surface, score_rect)

        high_score_surface = font.render(f'High score: {int(HIGH_SCORE)}', True, (255, 255, 255))
        high_score_rect = high_score_surface.get_rect(center=(SCREEN_WIDTH // 2, 150))
        screen.blit(high_score_surface, high_score_rect)

def update_score():
    global SCORE, HIGH_SCORE
    SCORE += 0.1
    if SCORE > HIGH_SCORE:
        HIGH_SCORE = SCORE

# Game variables
game_active = True
pipe_heights = [200, 300, 400]
bird_downflap = pygame.image.load('assets/yellowbird-downflap.png').convert_alpha()
bird_midflap = pygame.image.load('assets/yellowbird-midflap.png').convert_alpha()
bird_upflap = pygame.image.load('assets/yellowbird-upflap.png').convert_alpha()
bird_frames = [bird_downflap, bird_midflap, bird_upflap]
bird_index = 0
bird_surface = bird_frames[bird_index]
bird_rect = bird_surface.get_rect(center=(SCREEN_WIDTH // 5, SCREEN_HEIGHT // 2))
BIRDFLAP = pygame.USEREVENT + 1
pygame.time.set_timer(BIRDFLAP, 200)

# Game loop
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_active:
                bird.flap()
            if event.key == pygame.K_SPACE and not game_active:
                game_active = True
                pipe_list.clear()
                bird.rect.center = (SCREEN_WIDTH // 5, SCREEN_HEIGHT // 2)
                bird.bird_movement = 0
                SCORE = 0
        if event.type == pipe_timer and game_active:
            bottom_pipe, top_pipe = create_pipe()
            pipe_list.append(bottom_pipe)
            pipe_list.append(top_pipe)
        if event.type == BIRDFLAP:
            if bird_index < 2:
                bird_index += 1
            else:
                bird_index = 0
            bird_surface, bird_rect = bird_animation()

    # Background
    screen.blit(bg_surface, (0, 0))

    if game_active:
        # Pipes
        pipe_list = move_pipes(pipe_list)
        draw_pipes(pipe_list)
        # Bird
        bird.update()
        rotated_bird = rotate_bird(bird_surface)
        screen.blit(rotated_bird, bird.rect)
        check_collision(pipe_list)
        update_score()
        score_display('main_game')
    else:
        screen.blit(game_over_surface, game_over_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))
        score_display('game_over')

    # Floor
    floor_x_pos -= 1
    if floor_x_pos <= -SCREEN_WIDTH:
        floor_x_pos = 0
    screen.blit(floor_surface, (floor_x_pos, 450))
    screen.blit(floor_surface, (floor_x_pos + SCREEN_WIDTH, 450))
