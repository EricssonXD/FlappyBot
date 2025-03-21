TRAINING = False

if TRAINING:
    import os

    # os.environ["SDL_VIDEODRIVER"] = "dummy"  # Set before importing pygame


import time
import pygame
import flappy_bird
from dqn import Agent
import torch


def train():
    env = flappy_bird.Game(training_mode=True)
    agent = Agent(state_size=5, action_size=2)
    episodes = 2000
    batch_size = 64

    agent.load_checkpoint(
        "checkpoint.pth", epsilon=0.2
    )  # Load model, optimizer, and epsilon
    # agent.load_checkpoint("flappy_dqn_gen0.pth")  # Load model, optimizer, and epsilon

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # start_time = time.time()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(batch_size)
            pygame.event.pump()
            # end_time = time.time()
            # print(f"Step took: {end_time - start_time:.6f} seconds")

        print(f"Episode: {episode+1}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")

        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            agent.save_checkpoint("checkpoint.pth")

    torch.save(agent.model.state_dict(), "models/flappy_dqn.pth")


def test():
    env = flappy_bird.Game()
    agent = Agent(state_size=5, action_size=2)
    agent.model.load_state_dict(torch.load("models/flappy_dqn_gen0.pth"))
    # agent.load_checkpoint("checkpoint.pth")  # Load model, optimizer, and epsilon
    agent.epsilon = 0.0  # Disable exploration

    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, _, done = env.step(action)


if __name__ == "__main__":
    if TRAINING:
        train()  # Switch to test() to see the trained AI
        test()
    else:
        test()
        # flappy_bird.run()
