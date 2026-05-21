"""
Improved EMGNN model with the following enhancements over the benchmark:

1. Residual connections - skip connections between GNN layers to improve gradient flow
   and allow training deeper networks.
2. Normalisation options (norm_type parameter):
   - 'batch'  : BatchNorm1d  (default; NOTE: harmful on full-batch graphs — use 'none')
   - 'graph'  : GraphNorm    (designed for GNNs; normalises per-graph, not per-batch)
   - 'layer'  : LayerNorm    (normalises over feature dimension; works on any batch size)
   - 'none'   : no normalisation
3. Learnable network-importance weights - a softmax-normalised scalar per PPI network
   so the model can weight each network's contribution to the meta graph.
4. GraphSAGE (--sage) as an additional backbone option.
5. Label smoothing in the loss (optional) to reduce over-confidence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, GraphNorm
from torch_geometric.utils import add_self_loops, dropout_edge


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017) for class-imbalanced node classification.

    Works with log_softmax output (same interface as F.nll_loss).
    gamma=0 reduces to standard cross-entropy; gamma=2 is recommended.
    alpha sets the positive-class weight (0.75 means 3x weight for positives).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('alpha', torch.tensor([1 - alpha, alpha]))

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = log_probs.exp()
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha.to(log_probs.device)[targets]
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (focal_weight * nll).mean()


class HighLowPassSeparation(nn.Module):
    """
    Heterophily-aware gated fusion of low-pass (neighbourhood mean) and
    high-pass (self − neighbourhood mean) signals.

    PPI networks have low homophily — cancer genes' neighbours are mostly
    non-cancer genes — so standard GCN smoothing dilutes the discriminative
    signal.  This module lets the model learn per-node how much to rely on
    the smoothed vs. the residual (difference) representation.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid(),
        )

    def forward(self, x_original, x_smoothed):
        x_high = x_original - x_smoothed
        g = self.gate(torch.cat([x_smoothed, x_high], dim=-1))
        return g * x_smoothed + (1 - g) * x_high


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
        Deprecated shorthand — kept for backward compatibility.
        Equivalent to norm_type='batch'. If False, sets norm_type='none'.
    norm_type : str
        Normalisation to apply after each GNN layer. One of:
        'batch'  — BatchNorm1d (default; harmful on full-batch graphs)
        'graph'  — GraphNorm (GNN-aware, recommended alternative to BatchNorm)
        'layer'  — LayerNorm (works for any batch size)
        'none'   — no normalisation
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
        norm_type: str = 'batch',
        use_network_weights: bool = True,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.args = args
        self.use_residual = use_residual
        # Resolve norm_type: use_batchnorm=False overrides to 'none' for backward compat
        if not use_batchnorm:
            norm_type = 'none'
        self.norm_type = norm_type
        self.use_batchnorm = (norm_type != 'none')  # kept for backward compat checks
        self.use_network_weights = use_network_weights
        self.label_smoothing = label_smoothing
        self.n_layers = n_layers

        # ── Focal loss for class imbalance ─────────────────────────────────
        focal_gamma = getattr(args, 'focal_gamma', 0.0)
        focal_alpha = getattr(args, 'focal_alpha', 0.75)
        self.use_focal = focal_gamma > 0
        if self.use_focal:
            self.focal_loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

        alpha = getattr(args, 'alpha', 0.2)
        self.dropout = getattr(args, 'dropout', 0.5)
        self.drop_edge_rate = getattr(args, 'drop_edge_rate', 0.0)
        self.leakyrelu = nn.LeakyReLU(alpha)

        # ── Input projections ──────────────────────────────────────────────
        self.linear = nn.Linear(nfeat, hidden_channels)
        self.meta_linear = nn.Linear(nfeat, hidden_channels)

        # ── Per-layer GNN stack ────────────────────────────────────────────
        self.conv = nn.ModuleList([
            _build_conv(args, hidden_channels, hidden_channels)
            for _ in range(n_layers)
        ])

        # ── Heterophily-aware high/low-pass gating (optional) ─────────────
        self.heterophily_aware = getattr(args, 'heterophily_aware', False)
        if self.heterophily_aware:
            self.hl_gates = nn.ModuleList([
                HighLowPassSeparation(hidden_channels) for _ in range(n_layers)
            ])

        # ── Normalisation stack ────────────────────────────────────────────
        def _make_norm(channels):
            if norm_type == 'batch':
                return nn.BatchNorm1d(channels)
            elif norm_type == 'graph':
                return GraphNorm(channels)
            elif norm_type == 'layer':
                return nn.LayerNorm(channels)
            else:
                return None

        if norm_type != 'none':
            norms = [_make_norm(hidden_channels) for _ in range(n_layers)]
            self.bn_layers = nn.ModuleList(norms)
            self.meta_bn = _make_norm(hidden_channels)
        else:
            self.bn_layers = None
            self.meta_bn = None

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

        # ── Captum path: IG passes x_all = [input_nodes | meta_nodes] ────
        # Split so that subsequent steps only see input nodes in x,
        # matching the same logic as the original benchmark EMGNN.
        if captum and x.shape[0] > self.nb_nodes:
            meta_x = x[self.nb_nodes:]   # use captum-perturbed meta features
            x = x[:self.nb_nodes]        # keep only input-node portion

        # ── 1. Input projection ────────────────────────────────────────────
        x = self.leakyrelu(self.linear(x))
        meta_x_proj = self.leakyrelu(self.meta_linear(meta_x))

        # ── 2. Per-network importance weighting ───────────────────────────
        if self.network_weights is not None and hasattr(data, 'batch'):
            # softmax-normalise weights so they sum to 1
            w = F.softmax(self.network_weights, dim=0)
            # data.batch[i] tells us which graph node i belongs to
            node_w = w[data.batch]          # shape: (n_input_nodes,)
            x = x * node_w.unsqueeze(1)    # broadcast over feature dim

        # ── 3. Graph-level message passing with residual + normalisation ──────
        if self.training and self.drop_edge_rate > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.drop_edge_rate)

        for i in range(self.n_layers):
            identity = x
            x_conv = self.conv[i](x, edge_index)
            if self.bn_layers is not None:
                x_conv = self.bn_layers[i](x_conv)
            x_conv = self.leakyrelu(x_conv)

            if self.heterophily_aware:
                # Gated high/low-pass separation (replaces plain residual)
                x = self.hl_gates[i](identity, x_conv)
            else:
                x = x_conv
                if self.use_residual and i >= 1:
                    x = x + identity

            x = F.dropout(x, self.dropout, training=self.training)

        # ── 4. Meta-graph message passing ──────────────────────────────────
        x_all = torch.cat([x, meta_x_proj], dim=0)
        x_all = self.meta_gnn(x_all, meta_edge_index_use)
        if self.meta_bn is not None:
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
        Classification loss with optional focal weighting and label smoothing.

        When focal_gamma > 0, uses Focal Loss (Lin et al. 2017) to handle
        class imbalance by down-weighting easy negatives.
        Label smoothing blends the primary loss with a uniform-distribution term.
        """
        # Choose primary loss: focal or standard NLL
        if self.use_focal:
            primary = self.focal_loss_fn(output, labels)
        else:
            primary = F.nll_loss(output, labels)

        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            uniform = -output.mean()
            return (1.0 - eps) * primary + eps * uniform
        return primary
