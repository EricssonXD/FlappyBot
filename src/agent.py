import itertools
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from dqn import DQN

from config import (
    BATCH_SIZE,
    EPISODE_STOP_REWARD,
    EPSILON_DECAY,
    EPSILON_START,
    EPSILON_MIN,
    GAMMA,
    LEARNING_RATE,
    NETWORK_SYNC_FREQ,
)
import datetime

# import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque([], maxlen=5000)  # Experience replay buffer
        self.gamma = GAMMA  # Discount factor
        self.epsilon = EPSILON_START  # Exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.history_rewards = []
        self.history_epsilon = []

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def act(self, state):
        """
        State must be a tensor

        Outputs an action in the form of a tensor
        """
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            action = self.tensor(action, dtype=torch.int64)
        else:
            with torch.no_grad():
                action = self.model(state.unsqueeze(dim=0)).squeeze().argmax()
        return action

    def optimize(self, minibatch: list, model: DQN, target_model: DQN) -> None:
        """Optimizing the model"""
        states, actions, rewards, next_states, terminations = zip(*minibatch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminations)
                * self.gamma
                * target_model(next_states).max(dim=1)[0]
            )

        current_q = (
            model(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        )

        loss = torch.nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def tensor(self, state, dtype=torch.float):
        return torch.tensor(state, dtype=dtype, device=device)

    def plot(self, rewards_per_episode, epsilon_per_episode):

        fig, axs = plt.subplots(2, 1, figsize=(8, 10))

        axs[0].plot(rewards_per_episode)
        axs[0].set(xlabel="Episode", ylabel="Reward", title="Reward per episode")

        axs[1].plot(epsilon_per_episode)
        axs[1].set(xlabel="Episode", ylabel="Epsilon", title="Epsilon per episode")

        # fig.tight_layout()
        fig.savefig("metrics.png")
        plt.close(fig)

    def train(self, env):
        self.epsilon = EPSILON_START

        best_reward = -np.inf

        for episode in itertools.count():
            # start_time = time.time()
            terminated = False
            episode_reward = 0

            state = env.reset()
            state = self.tensor(state)

            step_counter = 0

            while not terminated and episode_reward < EPISODE_STOP_REWARD:
                action = self.act(state)

                next_state, reward, terminated = env.step(action.item())
                episode_reward += reward

                next_state = self.tensor(next_state)
                reward = self.tensor(reward)

                self.remember(state, action, reward, next_state, terminated)

                step_counter += 1

                state = next_state

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            self.history_rewards.append(episode_reward)
            self.history_epsilon.append(self.epsilon)

            if episode % 100 == 0:
                self.plot(self.history_rewards, self.history_epsilon)

            if len(self.memory) > BATCH_SIZE:

                minibatch = random.sample(self.memory, BATCH_SIZE)

                self.optimize(minibatch, self.model, self.target_model)

                if step_counter > NETWORK_SYNC_FREQ:
                    self.target_model.load_state_dict(self.model.state_dict())
                    step_counter = 0

            if episode_reward > best_reward:
                log_message = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Episode: {episode+1}, Reward: {episode_reward}, Score: {env.score}"
                print(log_message)

                with open("logs.txt", "a") as f:
                    f.write(log_message + "\n")

                torch.save(self.model.state_dict(), "models/flappy_dqn.pth")
                best_reward = episode_reward
            # end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")
        # torch.save(agent.model.state_dict(), "models/flappy_dqn.pth")
