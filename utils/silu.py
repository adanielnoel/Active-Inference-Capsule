import torch.nn as nn
import torch

"""
Implementation of the SiLU activation function for neural networks
[1] Elfwing, S., Uchibe, E., & Doya, K. (2018). Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107(2015), 3–11. https://doi.org/10.1016/j.neunet.2017.12.012
"""


class _SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# The SiLu activation (also known as Swish) was introduced in Pytorch 1.8. If not available, provide own implementation
if hasattr(torch.nn, 'SiLU'):
    SiLU = torch.nn.SiLU
else:
    SiLU = _SiLU
