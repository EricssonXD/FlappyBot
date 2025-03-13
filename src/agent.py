import itertools
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random

from config import (
    BATCH_SIZE,
    EPSILON_DECAY,
    EPSILON_START,
    EPSILON_MIN,
    GAMMA,
    LEARNING_RATE,
)
from dqn import DQN


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
        self.model_target = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def act(self, state):
        """
        State must be a tensor

        Outputs an action in the form of a tensor
        """
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            return self.tensor(action, dtype=torch.int8)

        with torch.no_grad():
            return self.model(state.unsqueeze(dim=0)).squeeze().argmax()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, terminated in minibatch:
            target = reward
            if not terminated:
                next_state = torch.FloatTensor(next_state)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()
            state = torch.FloatTensor(state)
            predicted_target = self.model(state)[action]
            loss = torch.nn.functional.mse_loss(
                predicted_target, torch.tensor(target, dtype=torch.float)
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def tensor(self, state, dtype=torch.float64):
        return torch.tensor(state, dtype=dtype, device=device)

    def train(self, env):
        for episode in itertools.count():
            terminated = False
            total_reward = 0

            state = env.reset()
            state = self.tensor(state)

            while not terminated:
                action = self.act(state)
                next_state, reward, terminated = env.step(action.item())
                total_reward += reward

                next_state = self.tensor(next_state)
                reward = self.tensor(reward)

                self.remember(state, action, reward, next_state, terminated)
                state = next_state
                self.replay(BATCH_SIZE)

                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            print(
                f"Episode: {episode+1}, Score: {env.score}, Epsilon: {self.epsilon:.2f}"
            )

        # torch.save(agent.model.state_dict(), "models/flappy_dqn.pth")
