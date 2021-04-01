import torch
import torch.nn as nn
import torch.optim as optim

from utils.silu import SiLU


class BiasedModelBellman(nn.Module):
    def __init__(self, observation_dim, iterate_train=1, discount_factor=0.99):
        super(BiasedModelBellman, self).__init__()
        self.observation_dim = observation_dim
        self.iterate_train = iterate_train
        self.discount_factor = discount_factor
        self.nn = nn.Sequential(
            nn.Linear(observation_dim, observation_dim * 10),
            SiLU(),
            nn.Linear(observation_dim * 10, observation_dim),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.nn(x)

    def learn(self, y: torch.Tensor):
        seq_length = y.shape[0]
        for i in range(self.iterate_train):  # make multiple stochastic gradient descend steps, randomize order of examples each time
            with torch.no_grad():
                p_ = self(y)
                py = torch.roll(p_, shifts=(-1, 0), dims=(0, 1))
                py[-1, :] = 1.0
                disc = torch.pow(self.discount_factor, torch.arange(seq_length, dtype=torch.float32).flip(0))
                disc = disc.expand((self.observation_dim, seq_length)).transpose(0, 1)  # Copy the probs so that the targets have shape [len(observations), observation_dim]
                py = py * disc

            self.optimizer.zero_grad()
            pred_p = self(y)
            loss = nn.functional.mse_loss(pred_p, py)
            loss.backward()
            self.optimizer.step()
