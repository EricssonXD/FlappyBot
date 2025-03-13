import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random

from dqn import DQN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque([], maxlen=5000)  # Experience replay buffer
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.model = DQN(state_size, action_size).to(device)
        self.model_target = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
