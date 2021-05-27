import torch.nn as nn


"""
Maps a value from a range onto another
"""


class ValueMap(nn.Module):
    def __init__(self, in_min, in_max, out_min, out_max):
        super(ValueMap, self).__init__()
        self.in_min = in_min
        self.in_max = in_max
        self.out_min = out_min
        self.out_max = out_max
        self.in_width = in_max - in_min
        self.out_width = out_max - out_min

    def forward(self, value):
        return (value - self.in_min) / self.in_width * self.out_width + self.out_min

    def inverse(self, value):
        return (value - self.out_min) / self.out_width * self.in_width + self.in_min
