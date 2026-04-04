"""
Improved EMGNN model with the following enhancements over the benchmark:

1. Residual connections - skip connections between GNN layers to improve gradient flow
   and allow training deeper networks.
2. Batch normalization - stabilises training and acts as regularisation.
3. Learnable network-importance weights - a softmax-normalised scalar per PPI network
   so the model can weight each network's contribution to the meta graph.
4. GraphSAGE (--sage) as an additional backbone option.
5. Label smoothing in the loss (optional) to reduce over-confidence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.utils import add_self_loops


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _build_conv(args, in_channels: int, out_channels: int) -> nn.Module:
    """Instantiate a single graph-convolution layer based on CLI args."""
    if getattr(args, 'gcn', False):
        return GCNConv(in_channels, out_channels)
    if getattr(args, 'gat', False):
        return GATConv(in_channels, out_channels,
                       heads=getattr(args, 'nb_heads', 1), concat=False)
    if getattr(args, 'gin', False):
        return GINConv(nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(getattr(args, 'alpha', 0.2)),
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, out_channels),
        ))
    if getattr(args, 'sage', False):
        return SAGEConv(in_channels, out_channels)
    # default fallback
    return GCNConv(in_channels, out_channels)


# ──────────────────────────────────────────────────────────────────────────────
# Improved EMGNN
# ──────────────────────────────────────────────────────────────────────────────

class EMGNNImproved(torch.nn.Module):
    """
    Improved version of the EMGNN benchmark model.

    Key additions
    -------------
    use_residual : bool
        Add identity skip-connections after every GNN layer (depth ≥ 2).
    use_batchnorm : bool
        Apply BatchNorm1d after every GNN layer and after the meta-GNN.
    use_network_weights : bool
        Learn a softmax-normalised scalar weight for each input PPI network.
        The weight is applied to the initial node representation before
        meta-graph aggregation.
    args.sage : bool
        New backbone option that uses SAGEConv (GraphSAGE) instead of
        GCN / GAT / GIN.
    label_smoothing : float
        Epsilon for label smoothing in the NLL loss (0 = off).

    All other behaviour mirrors the original EMGNN so that experiment scripts
    are drop-in compatible.
    """

    def __init__(
        self,
        nfeat: int,
        hidden_channels: int,
        n_layers: int,
        nclass: int,
        meta_x=None,
        args=None,
        data=None,
        node2idx=None,
        use_residual: bool = True,
        use_batchnorm: bool = True,
        use_network_weights: bool = True,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.args = args
        self.use_residual = use_residual
        self.use_batchnorm = use_batchnorm
        self.use_network_weights = use_network_weights
        self.label_smoothing = label_smoothing
        self.n_layers = n_layers

        alpha = getattr(args, 'alpha', 0.2)
        self.dropout = getattr(args, 'dropout', 0.5)
        self.leakyrelu = nn.LeakyReLU(alpha)

        # ── Input projections ──────────────────────────────────────────────
        self.linear = nn.Linear(nfeat, hidden_channels)
        self.meta_linear = nn.Linear(nfeat, hidden_channels)

        # ── Per-layer GNN stack ────────────────────────────────────────────
        self.conv = nn.ModuleList([
            _build_conv(args, hidden_channels, hidden_channels)
            for _ in range(n_layers)
        ])

        # ── Batch-norm stack ───────────────────────────────────────────────
        if use_batchnorm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(hidden_channels) for _ in range(n_layers)
            ])
            self.meta_bn = nn.BatchNorm1d(hidden_channels)

        # ── Meta-graph GNN ─────────────────────────────────────────────────
        self.meta_gnn = _build_conv(args, hidden_channels, hidden_channels)

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Linear(hidden_channels, nclass)

        # ── Learnable per-network importance weights ───────────────────────
        if use_network_weights and data is not None:
            n_graphs = len(data.node_names)  # number of PPI graphs in batch
            self.network_weights = nn.Parameter(torch.zeros(n_graphs))
        else:
            self.network_weights = None

        # ── Build meta-graph edge index (same logic as original EMGNN) ─────
        x = data.x.float()
        self.nb_nodes = x.shape[0]
        node_names = np.concatenate(data.node_names, axis=0)
        meta_src, meta_dst = [], []
        for i, node in enumerate(node_names):
            meta_src.append(i)
            meta_dst.append(node2idx[tuple(node)] + x.shape[0])
        self.meta_edge_index = torch.tensor([meta_src, meta_dst])
        self.meta_edge_index, _ = add_self_loops(self.meta_edge_index)
        self.meta_x = meta_x  # will be moved to device in forward

    # ──────────────────────────────────────────────────────────────────────────

    def _get_device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        edge_index,
        data,
        meta_edge_index=None,
        explain_x=None,
        captum: bool = False,
        explain: bool = False,
        edge_weight=None,
    ):
        device = self._get_device()
        meta_edge_index_use = (
            meta_edge_index if meta_edge_index is not None
            else self.meta_edge_index.to(device)
        )
        meta_x = self.meta_x.to(device)

        # ── 1. Input projection ────────────────────────────────────────────
        x = self.leakyrelu(self.linear(x))
        meta_x_proj = self.leakyrelu(self.meta_linear(meta_x))

        # ── 2. Per-network importance weighting ───────────────────────────
        if self.network_weights is not None and hasattr(data, 'batch'):
            # softmax-normalise weights so they sum to 1
            w = F.softmax(self.network_weights, dim=0)
            # data.batch[i] tells us which graph node i belongs to
            node_w = w[data.batch]          # shape: (total_nodes,)
            x = x * node_w.unsqueeze(1)    # broadcast over feature dim

        # ── 3. Graph-level message passing with residual + BN ──────────────
        for i in range(self.n_layers):
            identity = x
            x = self.conv[i](x, edge_index)
            if self.use_batchnorm:
                x = self.bn_layers[i](x)
            x = self.leakyrelu(x)
            # skip connection from layer 1 onward (dims always match here)
            if self.use_residual and i >= 1:
                x = x + identity
            x = F.dropout(x, self.dropout, training=self.training)

        # ── 4. Meta-graph message passing ──────────────────────────────────
        x_all = torch.cat([x, meta_x_proj], dim=0)
        x_all = self.meta_gnn(x_all, meta_edge_index_use)
        if self.use_batchnorm:
            x_all = self.meta_bn(x_all)
        x_all = self.leakyrelu(x_all)
        x_all = F.dropout(x_all, self.dropout, training=self.training)

        # ── 5. Classification ──────────────────────────────────────────────
        out = self.classifier(x_all)
        return F.log_softmax(out, dim=1)

    # ──────────────────────────────────────────────────────────────────────────
    # Loss helper (supports label smoothing)
    # ──────────────────────────────────────────────────────────────────────────

    def loss(self, output, labels):
        """
        Negative log-likelihood loss with optional label smoothing.

        Label smoothing replaces hard 0/1 targets with (eps/K, 1-eps+eps/K)
        which prevents the model from becoming over-confident and often
        improves calibration and generalisation.
        """
        if self.label_smoothing > 0.0:
            # output is already log_softmax (from forward)
            # Simple label-smoothing: blend NLL loss with uniform distribution loss
            eps = self.label_smoothing
            nll = F.nll_loss(output, labels)
            uniform = -output.mean()   # mean over all classes and samples
            return (1.0 - eps) * nll + eps * uniform
        return F.nll_loss(output, labels)
