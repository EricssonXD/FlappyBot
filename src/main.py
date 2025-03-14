from agent import Agent
from config import ACTION_SIZE, STATE_SIZE
from dqn import DQN
import flappy_bird
import pygame
import torch


def train():
    env = flappy_bird.Game(training_mode=True)
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    agent.train(env=env)


def test():
    env = flappy_bird.Game(training_mode=False)
    model = DQN(STATE_SIZE, ACTION_SIZE)
    model.load_state_dict(torch.load("models/flappy_dqn.pth"))
    model.eval()

    while True:
        state = env.reset()
        terminated = False
        while not terminated:
            pygame.event.pump()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            state, _, terminated = env.step(action)


def play():
    flappy_bird.run()


if __name__ == "__main__":
    # Ask the user if they want to train or test the model or play the game
    print("1. Train the model")
    print("2. Test the model")
    print("3. Play the game")

    choice = input("Enter your choice: ")
    if choice == "1":
        train()
    elif choice == "2":
        test()
    elif choice == "3":
        play()
    else:
        print("Invalid choice")
