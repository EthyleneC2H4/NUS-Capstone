"""
Baseline models for comparison.
  - GCNBaseline : single-network GCN (mirrors benchmark GCN class)
  - MLPBaseline : feature-only MLP (mirrors benchmark MLP class)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNBaseline(torch.nn.Module):
    """Simple GCN trained on a single PPI network (no meta-graph)."""

    def __init__(self, nfeat: int, hidden_channels: int, n_layers: int,
                 nclass: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.lins = nn.ModuleList([
            nn.Linear(nfeat, hidden_channels),
            nn.Linear(hidden_channels, nclass),
        ])
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels) for _ in range(n_layers)
        ])

    def forward(self, x, edge_index, data=None, edge_weight=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[0](x).relu()
        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight).relu()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return F.log_softmax(x, dim=1)


class MLPBaseline(torch.nn.Module):
    """3-layer MLP baseline that ignores graph structure."""

    def __init__(self, nfeat: int, hidden: int, nclass: int, alpha: float = 0.2):
        super().__init__()
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.fc1 = nn.Linear(nfeat, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, nclass)

    def forward(self, x, edge_index=None, data=None):
        x = F.dropout(self.leakyrelu(self.fc1(x)), training=self.training)
        x = F.dropout(self.leakyrelu(self.fc2(x)), training=self.training)
        return F.log_softmax(self.fc3(x), dim=1)
