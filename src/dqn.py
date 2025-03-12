import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# device = torch.device("cpu")


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
        self.alpha = 0.4  # Prioritization exponent
        # self.model = nn.DataParallel(DQN(state_size, action_size)).to(device)
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

    def save_checkpoint(self, filename="checkpoint.pth"):
        # Save model + optimizer + exploration rate
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename="checkpoints/checkpoint.pth"):
        try:
            checkpoint = torch.load(filename, map_location=device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.to(device)
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.epsilon = checkpoint["epsilon"]
            print(f"Loaded checkpoint from {filename}")
        except FileNotFoundError:
            print("No checkpoint found. Starting fresh.")

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
        # for index, (priority, state, action, reward, next_state, done) in enumerate(
        #     minibatch
        # ):
        #     # Compute target and loss
        #     target = reward
        #     if not done:
        #         try:
        #             next_state = next_state.clone().detach().requires_grad_(True)
        #         except:
        #             next_state = torch.tensor(
        #                 next_state, dtype=torch.float32, device=device
        #             )

        #         target = reward + self.gamma * torch.max(self.model(next_state)).item()

        #     try:
        #         state = state.clone().detach().requires_grad_(True)
        #     except:
        #         state = torch.tensor(state, dtype=torch.float32, device=device)

        #     predicted = self.model(state)[action]
        #     loss = torch.nn.functional.mse_loss(
        #         predicted, torch.tensor(target, dtype=torch.float32, device=device)
        #     )

        #     # Backpropagation
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()

        #     # Update priority with new error
        #     error = abs(predicted.item() - target)
        #     self.memory[indices[index]] = (
        #         error,
        #         state,
        #         action,
        #         reward,
        #         next_state,
        #         done,
        #     )

        # Create batch tensors
        # states = torch.stack([x[1] for x in minibatch], dim=0).to(device)
        states = torch.stack(
            [
                (
                    x[1].clone().detach()
                    if isinstance(x[1], torch.Tensor)
                    else torch.tensor(x[1], device=device).clone().detach()
                )
                for x in minibatch
            ],
            dim=0,
        )
        actions = torch.tensor(
            [x[2] for x in minibatch], dtype=torch.long, device=device
        )
        rewards = torch.tensor(
            [x[3] for x in minibatch], dtype=torch.float32, device=device
        )
        # next_states = torch.stack([x[4] for x in minibatch], dim=0).to(device)
        next_states = torch.stack(
            [
                (
                    x[4].clone().detach()
                    if isinstance(x[4], torch.Tensor)
                    else torch.tensor(x[4], device=device).clone().detach()
                )
                for x in minibatch
            ],
            dim=0,
        )
        # dones = torch.tensor([x[5] for x in minibatch], dtype=torch.bool, device=device)

        # Compute targets
        with torch.no_grad():
            next_state_values = self.model(next_states)
            targets = rewards + self.gamma * torch.max(next_state_values, dim=1)[0]
            # * (
            #     ~dones
            # )

        # Compute predictions and loss
        q_values = self.model(states)
        predictions = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = torch.nn.functional.mse_loss(predictions, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        with torch.no_grad():
            errors = torch.abs(predictions - targets)
            for i, idx in enumerate(indices):
                self.memory[idx] = (
                    errors[i].item(),
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i],
                    # dones[i],
                )

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
