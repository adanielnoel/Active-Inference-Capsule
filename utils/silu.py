import torch.nn as nn
import torch


class _SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# The SiLu activation (also known as Swish) was introduced in Pytorch 1.8. If not available, provide own implementation
if hasattr(torch.nn, 'SiLU'):
    SiLU = torch.nn.SiLU
else:
    SiLU = _SiLU
