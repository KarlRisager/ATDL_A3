import torch
from torch import nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        torch.manual_seed(1234567)
        self.layers = nn.Sequential(
            GATConv(in_channels, hidden_channels, heads=heads),
            nn.ReLU(),
            GATConv(hidden_channels * heads, out_channels, heads=heads)
        )


    def forward(self, x):
        x = self.layers(x)
        return x