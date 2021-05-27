import torch
import torch.nn as nn
import torch.optim as optim

from utils.silu import SiLU


class PriorModelBellman(nn.Module):
    def __init__(self, observation_dim, learning_rate=0.001, iterate_train=1, discount_factor=0.99):
        super(PriorModelBellman, self).__init__()
        self.observation_dim = observation_dim
        self.learning_rate = learning_rate
        self.iterate_train = iterate_train
        self.discount_factor = discount_factor
        self.nn = nn.Sequential(
            nn.Linear(observation_dim, observation_dim * 20),
            SiLU(),
            nn.Linear(observation_dim * 20, observation_dim),
            nn.Tanh()
        )
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        self.observations = []
        self.rewards = []

    def reset_states(self):
        self.observations = []
        self.rewards = []

    def forward(self, y):
        return self.nn(y)

    def extrinsic_kl(self, y):
        return 1.0 - self.forward(y) # map from [-1, 1] to [2, 0]

    def learn(self, y: torch.Tensor, r: float):
        self.observations.append(y)
        self.rewards.append(r)
        if abs(r) > 0.1:
            observations = torch.stack(self.observations)
            rewards = torch.tensor(self.rewards)
            rewards = rewards.expand((self.observation_dim, observations.shape[0])).transpose(0, 1)
            for i in range(self.iterate_train):
                disc = torch.pow(self.discount_factor, torch.arange(observations.shape[0], dtype=torch.float32).flip(0))
                disc = disc.expand((self.observation_dim, observations.shape[0])).transpose(0, 1)  # Apply same for each observation dim
                forward_ = self.forward(observations)
                with torch.no_grad():
                    pred_next_v = torch.cat([forward_[1:], torch.zeros((1, self.observation_dim))], dim=0)
                    r_ = rewards + disc * pred_next_v

                self.optimizer.zero_grad()
                for g in self.optimizer.param_groups:
                    g['lr'] = self.learning_rate * abs(r)
                loss = nn.functional.mse_loss(forward_, r_)
                loss.backward()
                self.optimizer.step()
