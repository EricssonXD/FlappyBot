import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=256,
        hidden_size2=128,
        use_dueling=True,
    ):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.use_dueling = use_dueling

        if self.use_dueling:
            self.fc_value = nn.Linear(hidden_size2, 256)
            self.value = nn.Linear(256, 1)

            self.fc_advantage = nn.Linear(hidden_size2, 256)
            self.advantage = nn.Linear(256, output_size)
        else:
            self.output = nn.Linear(hidden_size2, output_size)

    def forward(self, x) -> torch.Tensor:

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        if self.use_dueling:
            V = torch.relu(self.fc_value(x))
            A = torch.relu(self.fc_advantage(x))

            V: torch.Tensor = self.value(V)
            A: torch.Tensor = self.advantage(A)

            return V + A - A.mean(dim=1, keepdim=True)

        return self.output(x)
