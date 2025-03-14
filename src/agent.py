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
    DISCOUNT_FACTOR,
    LEARNING_RATE,
    TARGET_UPDATE_FREQ,
    USE_DOUBLE_DQN,
)
import datetime
import time


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Agent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.memory: deque = deque([], maxlen=5000)  # Experience replay buffer
        self.gamma: float = DISCOUNT_FACTOR  # Discount factor
        self.epsilon: float = EPSILON_START  # Exploration rate
        self.epsilon_min: float = EPSILON_MIN
        self.epsilon_decay: float = EPSILON_DECAY
        self.model: DQN = DQN(state_size, action_size).to(device)
        self.target_model: DQN = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.history_rewards: list = []
        self.history_epsilon: list = []
        self.history_score: list = []
        self.history_score_avg: list = []

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def act(self, state: torch.Tensor) -> torch.Tensor:
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
            if USE_DOUBLE_DQN:
                next_state_actions = model(next_states).argmax(dim=1)
                target_q = (
                    rewards
                    + (1 - terminations)
                    * self.gamma
                    * target_model(next_states)
                    .gather(dim=1, index=next_state_actions.unsqueeze(dim=1))
                    .squeeze()
                )
            else:
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

    def tensor(self, state, dtype: torch.dtype = torch.float) -> torch.Tensor:
        return torch.tensor(state, dtype=dtype, device=device)

    def plot(
        self,
        rewards_per_episode: list,
        epsilon_per_episode: list,
        average_score_per_100_episode: list,
    ):

        fig, axs = plt.subplots(4, 1, figsize=(8, 10))

        axs[0].plot(rewards_per_episode)
        axs[0].set(xlabel="Episode", ylabel="Reward", title="Reward per episode")

        axs[1].plot(epsilon_per_episode)
        axs[1].set(xlabel="Episode", ylabel="Epsilon", title="Epsilon per episode")

        axs[2].plot(self.history_score)
        axs[2].set(xlabel="Episode", ylabel="Score", title="Score per episode")

        x_values = list(range(100, 100 * len(average_score_per_100_episode) + 1, 100))
        axs[3].plot(x_values, average_score_per_100_episode)
        axs[3].set(
            xlabel="Episode",
            ylabel="Mean Score",
            title="Mean Score per 100 Episodes",
        )

        # fig.tight_layout()
        fig.savefig("metrics.png")
        plt.close(fig)

    def train(self, env):
        self.epsilon = EPSILON_START

        best_reward = -np.inf

        starttime = time.time()

        for episode in itertools.count():
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
            self.history_score.append(env.score)

            if episode % 100 == 0:
                endtime = time.time()
                average_score = sum(self.history_score[-100:]) / 100
                print(
                    f"Took {endtime - starttime} seconds: min: {min(self.history_score[-100:])}, max: {max(self.history_score[-100:])}, avg: {average_score}"
                )
                self.history_score_avg.append(average_score)

                self.plot(self.history_rewards, self.history_epsilon, average_score)
                starttime = time.time()

            if len(self.memory) > BATCH_SIZE:

                minibatch = random.sample(self.memory, BATCH_SIZE)

                self.optimize(minibatch, self.model, self.target_model)

                if step_counter > TARGET_UPDATE_FREQ:
                    self.target_model.load_state_dict(self.model.state_dict())
                    step_counter = 0

            if episode_reward > best_reward:
                log_message = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Episode: {episode+1}, Reward: {episode_reward}, Score: {env.score}"
                print(log_message)

                with open("logs.txt", "a") as f:
                    f.write(log_message + "\n")

                torch.save(self.model.state_dict(), "models/flappy_dqn.pth")
                best_reward = episode_reward
