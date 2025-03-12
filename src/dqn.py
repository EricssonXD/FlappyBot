import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.6  # Prioritization exponent
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        # Store new experiences with max priority (to ensure they get sampled)
        max_priority = max([exp[0] for exp in self.memory], default=1.0)
        self.memory.append((max_priority, state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Calculate sampling probabilities
        priorities = np.array([exp[0] for exp in self.memory])
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Select batch based on priorities
        indices = np.random.choice(len(self.memory), size=batch_size, p=probs)
        minibatch = [self.memory[i] for i in indices]

        # Train and update priorities
        for index, (priority, state, action, reward, next_state, done) in enumerate(
            minibatch
        ):
            # Compute target and loss
            target = reward
            if not done:
                next_state = torch.tensor(
                    next_state, dtype=torch.float32, device=device
                )
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            state = torch.tensor(state, dtype=torch.float32, device=device)
            predicted = self.model(state)[action]
            loss = torch.nn.functional.mse_loss(
                predicted, torch.tensor(target, dtype=torch.float32, device=device)
            )

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update priority with new error
            error = abs(predicted.item() - target)
            self.memory[indices[index]] = (
                error,
                state,
                action,
                reward,
                next_state,
                done,
            )

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
