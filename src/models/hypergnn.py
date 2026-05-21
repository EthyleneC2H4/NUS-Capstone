"""
Hypergraph Neural Network for pathway-level message passing.

Implements the two-stage message passing of HGNN (Feng et al., AAAI 2019):
  node → hyperedge → node

The hypergraph encodes functional annotations (GO terms, KEGG pathways)
as hyperedges. This provides pathway-level structural priors complementary
to the PPI graph topology.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperGNNLayer(nn.Module):
    """
    Single hypergraph convolution layer.

    X' = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X Theta

    where H is the incidence matrix, D_v/D_e are degree matrices.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (n_nodes, in_channels)
        H : (n_nodes, n_hyperedges) — incidence matrix
        """
        D_v = H.sum(dim=1, keepdim=True).clamp(min=1)   # node degree
        D_e = H.sum(dim=0, keepdim=True).clamp(min=1)   # hyperedge degree
        D_v_inv_sqrt = D_v.pow(-0.5)
        D_e_inv = D_e.pow(-1)

        # Stage 1: node → hyperedge (aggregate)
        x_norm = x * D_v_inv_sqrt
        edge_feat = torch.mm(H.t(), x_norm)
        edge_feat = edge_feat * D_e_inv.t()

        # Stage 2: hyperedge → node (distribute)
        x_out = torch.mm(H, edge_feat)
        x_out = x_out * D_v_inv_sqrt

        return self.linear(x_out)


class HyperGNNEncoder(nn.Module):
    """Multi-layer hypergraph encoder with residual connections."""

    def __init__(self, in_channels: int, hidden: int,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.input_linear = nn.Linear(in_channels, hidden)
        self.layers = nn.ModuleList([
            HyperGNNLayer(hidden, hidden) for _ in range(n_layers)
        ])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.input_linear(x), 0.2)
        for layer in self.layers:
            identity = x
            x = layer(x, H)
            x = F.leaky_relu(x, 0.2)
            x = x + identity  # residual
            x = F.dropout(x, self.dropout, training=self.training)
        return x
