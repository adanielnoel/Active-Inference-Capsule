import torch
import torch.nn as nn
import torch.distributions as distr
import torch.optim as optim

from models.vae import VAE
from utils.silu import SiLU


class Enc(nn.Module):
    def __init__(self, observation_dim, latent_dim):
        super(Enc, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(observation_dim, observation_dim * 10),
            SiLU()
        )
        self.c1 = nn.Linear(latent_dim * 10, latent_dim)
        self.c2 = nn.Linear(latent_dim * 10, latent_dim)

    def forward(self, y):
        e = self.enc(y)
        return self.c1(e), nn.functional.softplus(self.c2(e)) + 1e-6


class Dec(nn.Module):
    def __init__(self, observation_dim, latent_dim):
        super(Dec, self).__init__()
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 10),
            SiLU()
        )
        self.c1 = nn.Linear(observation_dim * 10, observation_dim)

    def forward(self, x):
        d = self.dec(x)
        return self.c1(d), torch.tensor(0.02).to(x.device)


class DenseObservation_VAE(VAE):
    def __init__(self, observation_dim, latent_dim):
        super(DenseObservation_VAE, self).__init__(
            prior_dist=distr.Normal,
            likelihood_dist=distr.Normal,
            post_dist=distr.Normal,
            enc=Enc(observation_dim, latent_dim),
            dec=Dec(observation_dim, latent_dim),
            observation_dim=observation_dim,
            latent_dim=latent_dim
        )
        self._px_params = [torch.zeros(1, latent_dim),  # mu
                           torch.ones(1, latent_dim) * 0.3]  # var
        self.modelName = 'Dense observation VAE'
        self.data_shape = [-1, observation_dim]
        self.llik_scaling = latent_dim / observation_dim  # Scale factor for the log-likelihood in the loss function
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def learn(self, y):
        self.optimizer.zero_grad()
        VFE, qx_y = self.loss(y)
        VFE.mean().backward()
        self.optimizer.step()
        qx_y.loc.detach_()
        qx_y.scale.detach_()
        return VFE.detach(), qx_y
