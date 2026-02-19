import torch
from torch import nn
from torch.nn import functional as F

class LIA1D(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, f, kernel_size=1)
        self.softpool = nn.AvgPool1d(kernel_size=7, stride=3, padding=0)
        self.conv2 = nn.Conv1d(f, f, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(f, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Sequential(nn.Sigmoid())

        kernel = torch.ones(3) / 3.0
        self.register_buffer('smooth_kernel', kernel.view(1, 1, 3))  # shape: [1, 1, 3]

    def forward(self, x):
        # x: [B, L, D] → [B, D, L]
        x = x.transpose(1, 2)  # [B, D, L]
        g = self.gate(x[:, :1])  # Gate Control

        w = self.conv1(x)  # [B, f, L]
        w = self.softpool(w)  # ↓
        w = self.conv2(w)  # ↓
        w = self.conv3(w)  # [B, D, l']
        w = self.sigmoid(w)

        # Nearest Neighbor Interpolation (Reproducible)+Smooth Convolution
        w = F.interpolate(w, size=x.size(2), mode='nearest')
        w = F.conv1d(w, self.smooth_kernel.repeat(w.size(1), 1, 1), padding=1, groups=w.size(1))

        out = x * w * g
        return out.transpose(1, 2)  # → [B, L, D]