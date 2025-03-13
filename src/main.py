# def main():
#     flappy_bird.run()

# # Test the game manually
# if __name__ == "__main__":
#     main()

import itertools
from agent import Agent
from config import ACTION_SIZE, STATE_SIZE
import flappy_bird
import torch


def train():
    env = flappy_bird.Game(training_mode=True)
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    batch_size = 32

    for episode in itertools.count():
        state = env.reset()
        terminated = False
        total_reward = 0

        while not terminated:
            action = agent.act(state)
            next_state, reward, terminated = env.step(action)
            agent.remember(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            agent.replay(batch_size)

        print(f"Episode: {episode+1}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")

    # torch.save(agent.model.state_dict(), "models/flappy_dqn.pth")


def test():
    env = flappy_bird.Game(training_mode=False)
    agent = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    agent.model.load_state_dict(torch.load("models/flappy_dqn.pth"))
    agent.epsilon = 0.0  # Disable exploration

    state = env.reset()
    terminated = False
    while not terminated:
        action = agent.act(state)
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
