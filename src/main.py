TRAINING = True

if TRAINING:
    import os

    os.environ["SDL_VIDEODRIVER"] = "dummy"  # Set before importing pygame
import time  # noqa: E402
import pygame  # noqa: E402
import flappy_bird  # noqa: E402
from dqn import Agent  # noqa: E402
import torch  # noqa: E402


def train():
    env = flappy_bird.Game(training_mode=True)
    agent = Agent(state_size=5, action_size=2)
    episodes = 1000
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            start_time = time.time()

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(batch_size)
            pygame.event.pump()

            end_time = time.time()
            print(f"Step took: {end_time - start_time:.6f} seconds")

        print(f"Episode: {episode+1}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")

    torch.save(agent.model.state_dict(), "models/flappy_dqn.pth")


def test():
    env = flappy_bird.Game()
    agent = Agent(state_size=5, action_size=2)
    agent.model.load_state_dict(torch.load("models/flappy_dqn.pth"))
    agent.epsilon = 0.0  # Disable exploration

    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, _, done = env.step(action)


if __name__ == "__main__":
    if TRAINING:
        train()  # Switch to test() to see the trained AI
    else:
        # test()
        flappy_bird.run()
