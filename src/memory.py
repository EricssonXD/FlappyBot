from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, maxlen=100000, seed=0):
        self.buffer = deque([], maxlen=maxlen)

    def append(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        return sample

    def __len__(self):
        return len(self.buffer)


class PriorityReplayBuffer:
    def __init__(self, maxlen=100000, seed=0):
        self.buffer = deque([], maxlen=maxlen)
        self.max_priority = 1.0
        np.random.seed(seed)

    def __len__(self):
        return len(self.buffer)

    def append(self, data):
        # Append new data with maximum priority so that new experiences
        # are likely to be sampled at least once.
        self.buffer.append((self.max_priority, data))

    def sample(self, batch_size, alpha=0.6):
        # Compute sampling probabilities proportional to priority^alpha.
        priorities = np.array([item[0] for item in self.buffer], dtype=np.float64)
        probabilities = priorities**alpha
        probabilities /= probabilities.sum()

        # Sample batch indices based on computed probabilities.
        indices = np.random.choice(
            len(self.buffer), batch_size, p=probabilities, replace=False
        )
        samples = [self.buffer[idx][1] for idx in indices]

        # Also return the sampling probabilities for IS weight calculation if needed.
        sampled_probs = probabilities[indices]
        return samples, indices, sampled_probs

    def update_priorities(self, indices, new_priorities):
        # Update priorities for the sampled experiences.
        for idx, priority in zip(indices, new_priorities):
            # Ensure the new priority is a float and update max_priority if needed.
            priority = float(priority)
            self.max_priority = max(self.max_priority, priority)
            # Update the tuple in the buffer.
            data = self.buffer[idx][1]
            self.buffer[idx] = (priority, data)
