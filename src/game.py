import pygame

class Game:
    def __init__(self):
        pygame.init()
        self.WIDTH = 800
        self.HEIGHT = 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Flappy Bot")
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.running = True

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        pass

    def render(self):
        self.screen.fill(self.WHITE)
        pygame.display.flip()

    def quit(self):
        pygame.quit()
