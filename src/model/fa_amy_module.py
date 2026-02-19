import torch
from torch import nn
from model.layers.bi_tcn import BiTCN
from model.layers.multi_head_attention import MultiHeadSelfAttention
from model.layers.lia_1d import LIA1D

class FAAmyModule(nn.Module):
    def __init__(self):
        super(FAAmyModule, self).__init__()
        self.bitcn = BiTCN(input_dim=1152, num_channels=[512, 256, 64], kernel_size=5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)
        self.drop = nn.Dropout(0.3)
        self.att1 = MultiHeadSelfAttention(input_dim=128, num_heads=4)
        self.att2 = LIA1D(128, f=64)
        self.relu = nn.LeakyReLU()

    def forward(self, prot0):
        prot2 = self.bitcn(prot0.permute(0, 2, 1))  # [B, 128, L]
        prot2 = prot2.permute(0, 2, 1)
        atten1 = self.att1(prot2)
        atten2 = self.att2(prot2)
        fused = 0.5 * atten1 + 0.5 * atten2
        pooled = prot2 + fused
        pooled = torch.mean(pooled, dim=1)
        x = self.fc2(pooled)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x
