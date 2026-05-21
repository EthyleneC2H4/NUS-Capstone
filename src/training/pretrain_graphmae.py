"""
GraphMAE self-supervised pretraining (Hou et al., KDD 2022).

Masks node features → GCN encode → GCN decode → reconstruct via scaled
cosine error.  The pretrained encoder weights can then be loaded into
EMGNNImproved for fine-tuning on the labelled cancer-gene task.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class _Encoder(nn.Module):
    """GCN encoder matching EMGNNImproved's conv stack for weight transfer."""

    def __init__(self, nfeat: int, hidden: int, n_layers: int, dropout: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(nfeat, hidden)
        self.convs = nn.ModuleList(
            [GCNConv(hidden, hidden) for _ in range(n_layers)]
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.linear(x), 0.2)
        for i, conv in enumerate(self.convs):
            identity = x
            x = F.leaky_relu(conv(x, edge_index), 0.2)
            if i >= 1:
                x = x + identity
            x = F.dropout(x, self.dropout, training=self.training)
        return x


class _Decoder(nn.Module):
    """Single-layer GCN decoder to reconstruct masked features."""

    def __init__(self, hidden: int, nfeat: int):
        super().__init__()
        self.conv = GCNConv(hidden, hidden)
        self.linear = nn.Linear(hidden, nfeat)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv(x, edge_index), 0.2)
        return self.linear(x)


class GraphMAE(nn.Module):
    def __init__(self, nfeat: int, hidden: int, n_layers: int,
                 mask_ratio: float = 0.5):
        super().__init__()
        self.encoder = _Encoder(nfeat, hidden, n_layers)
        self.decoder = _Decoder(hidden, nfeat)
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, nfeat))
        nn.init.xavier_uniform_(self.mask_token)

    def forward(self, x, edge_index):
        n = x.size(0)
        n_mask = int(n * self.mask_ratio)
        perm = torch.randperm(n, device=x.device)
        mask_idx = perm[:n_mask]

        x_masked = x.clone()
        x_masked[mask_idx] = self.mask_token

        z = self.encoder(x_masked, edge_index)
        x_recon = self.decoder(z, edge_index)

        # Scaled cosine error on masked nodes only
        target = x[mask_idx]
        pred = x_recon[mask_idx]
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        loss = (1 - cos_sim).pow(2).mean()
        return loss


def pretrain_graphmae(loader, info, args, device,
                      pretrain_epochs: int = 200,
                      pretrain_lr: float = 1e-3) -> dict:
    """
    Run GraphMAE pretraining on the full PPI graph (all nodes, no labels).

    Returns the encoder state_dict for loading into EMGNNImproved.
    """
    batch = info['batch']
    nfeat = batch.x.shape[1]

    mae = GraphMAE(
        nfeat=nfeat,
        hidden=args.hidden,
        n_layers=args.n_layers,
        mask_ratio=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(mae.parameters(), lr=pretrain_lr, weight_decay=1e-5)
    data = next(iter(loader)).to(device)

    mae.train()
    for epoch in range(pretrain_epochs):
        optimizer.zero_grad()
        loss = mae(data.x.float(), data.edge_index)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"  [GraphMAE] Epoch {epoch:3d}  loss={loss.item():.4f}")

    print(f"  [GraphMAE] Done — final loss={loss.item():.4f}")
    return mae.encoder.state_dict()


def load_pretrained_weights(model, encoder_state: dict) -> int:
    """
    Map pretrained encoder weights into an EMGNNImproved model.

    Returns the number of parameters successfully transferred.

    Mapping: encoder.linear  → model.linear
             encoder.convs.i → model.conv.i
    """
    model_state = model.state_dict()
    loaded = 0
    for key, value in encoder_state.items():
        mapped = key.replace('convs.', 'conv.')
        if mapped in model_state and model_state[mapped].shape == value.shape:
            model_state[mapped] = value
            loaded += 1
    model.load_state_dict(model_state)
    return loaded
