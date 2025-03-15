from collections import deque
import random


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
