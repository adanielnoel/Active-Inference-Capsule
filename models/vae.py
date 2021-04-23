# Base VAE class definition

import torch
import torch.nn as nn
import torch.distributions as distr

"""
Base class for variational autoencoders
Adapted from https://github.com/iffsid/mmvae
"""


class VAE(nn.Module):
    def __init__(self, enc, dec, observation_dim, latent_dim):
        super(VAE, self).__init__()
        self.enc = enc
        self.dec = dec
        self._latent_dim = latent_dim
        self._observation_dim = observation_dim
        self.modelName = None
        self._px_params = None  # defined in subclass
        self.llik_scaling = 1.0
        self.data_shape = None  # defined in subclass, e.g., [-1, 3, 420, 600]

    @property
    def px(self):
        return distr.Normal(*self._px_params)

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    def forward(self, y) -> (distr.Distribution, distr.Distribution, torch.Tensor):
        qx_y_params = self.enc(y)
        qx_y = distr.Normal(*qx_y_params)
        x_samples = qx_y.rsample()
        py_x = distr.Normal(*self.dec(x_samples))
        return qx_y, py_x, x_samples

    def loss(self, y):
        qx_y, py_x, latents = self(y)
        lpy_x = py_x.log_prob(y) * self.llik_scaling
        kld = distr.kl_divergence(qx_y, self.px)
        VFEs = kld.sum(-1) - lpy_x.sum(-1)  # Computes variational free energy of each observation
        return VFEs, qx_y

    def learn(self, y):
        pass

    def generate(self, n_samples):
        latents = self.px.rsample(torch.Size([n_samples]))
        return self.decode(latents)

    def reconstruct(self, data):
        latents = self.infer(data)
        recon = self.decode(latents)
        return recon

    def infer_density(self, data):
        return distr.Normal(*self.enc(data))

    def decode_density(self, latents):
        return distr.Normal(*self.dec(latents))

    def decode(self, latents):
        recon = self.decode_density(latents).mean
        return recon

    def infer(self, data):
        latents = self.infer_density(data).mean
        return latents
