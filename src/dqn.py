import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=256,
        hidden_size2=128,
        use_dueling=True,
        use_noisy_net=True,
        device="cpu",
    ):
        super(DQN, self).__init__()
        self.use_dueling = use_dueling

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)

        if self.use_dueling:
            self.fc_value = nn.Linear(hidden_size2, 256)
            self.value = nn.Linear(256, 1)

            if use_noisy_net:
                self.fc_advantage = NoisyLayer(hidden_size2, 256, device=device)
                self.advantage = NoisyLayer(256, output_size, device=device)
            else:
                self.fc_advantage = nn.Linear(hidden_size2, 256)
                self.advantage = nn.Linear(256, output_size)
        else:
            if use_noisy_net:
                self.output = NoisyLayer(hidden_size2, output_size, device=device)
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


class NoisyLayer(nn.Module):
    def __init__(self, input_features, output_features, sigma=0.5, device="cpu"):
        super().__init__()  # Fixed super() call
        self.device = device
        self.input_features = input_features
        self.output_features = output_features

        self.sigma = sigma
        self.bound = input_features ** (-0.5)

        # Learnable parameters
        self.mu_bias = nn.Parameter(torch.FloatTensor(output_features, device=device))
        self.sigma_bias = nn.Parameter(
            torch.FloatTensor(output_features, device=device)
        )
        self.mu_weight = nn.Parameter(
            torch.FloatTensor(output_features, input_features, device=device)
        )
        self.sigma_weight = nn.Parameter(
            torch.FloatTensor(output_features, input_features, device=device)
        )

        # Noise buffers
        self.register_buffer(
            "epsilon_input", torch.FloatTensor(input_features, device=device)
        )
        self.register_buffer(
            "epsilon_output", torch.FloatTensor(output_features, device=device)
        )

        self.parameter_initialization()
        self.sample_noise()

    def parameter_initialization(self):
        # Initialize mu and sigma
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.sigma_bias.data.fill_(self.sigma * self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)
        self.sigma_weight.data.fill_(self.sigma * self.bound)

    def forward(self, x: torch.Tensor, sample_noise: bool = True) -> torch.Tensor:
        if not self.training:
            # Use mu_weight and mu_bias during evaluation
            return F.linear(x, weight=self.mu_weight, bias=self.mu_bias)

        if sample_noise:
            self.sample_noise()

        # Combine mu and sigma with noise
        weight = (
            self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input)
            + self.mu_weight
        )
        bias = self.sigma_bias * self.epsilon_output + self.mu_bias
        return F.linear(x, weight=weight, bias=bias)

    def sample_noise(self):
        # Sample new noise for inputs and outputs
        self.epsilon_input = self.get_noise_tensor(self.input_features)
        self.epsilon_output = self.get_noise_tensor(self.output_features)

    def get_noise_tensor(self, features: int) -> torch.Tensor:
        # Factorized Gaussian noise
        noise = torch.randn(features)  # Standard Gaussian noise
        return torch.sign(noise) * torch.sqrt(torch.abs(noise))
