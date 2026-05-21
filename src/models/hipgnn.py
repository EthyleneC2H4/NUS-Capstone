"""
HIPGNN-inspired anomaly detection for cancer gene identification.

Reframes cancer gene prediction as graph anomaly detection by combining:
1. Spectral view: node energy distribution across Laplacian eigenvectors
   (cancer genes show anomalous high-frequency energy)
2. Spatial view: neighbourhood feature heterogeneity
   (cancer genes differ from their neighbours more than normal genes)

Used as an auxiliary objective in EMGNNImproved (multi-task learning):
    total_loss = classification_loss + lambda * anomaly_loss

Reference: HIPGNN (AAAI 2025) — adapted for PPI cancer gene detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SpectralAnomalyModule(nn.Module):
    """
    Spectral view: analyse each node's energy distribution across
    Laplacian eigenvectors. Cancer genes exhibit anomalous energy
    in high-frequency components (heterophily with neighbours).
    """

    def __init__(self, n_eigvecs: int, hidden: int):
        super().__init__()
        self.n_eigvecs = n_eigvecs
        self.energy_mlp = nn.Sequential(
            nn.Linear(n_eigvecs, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
        )

    def forward(self, eigvecs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eigvecs : (N, K) — Laplacian eigenvectors for each node
        """
        energy = eigvecs.abs()  # simplified energy distribution
        return self.energy_mlp(energy)


class SpatialAnomalyModule(nn.Module):
    """
    Spatial view: measure self-vs-neighbourhood feature heterogeneity.
    Cancer genes' features differ significantly from their neighbours.
    """

    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.conv_mean = GCNConv(in_channels, hidden, add_self_loops=False)
        self.diff_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
        )
        self.proj = nn.Linear(in_channels, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_self = self.proj(x)
        x_neighbour_mean = self.conv_mean(x, edge_index)
        diff = x_self - x_neighbour_mean
        return self.diff_mlp(torch.cat([x_neighbour_mean, diff], dim=-1))


class HIPGNNAuxHead(nn.Module):
    """
    Auxiliary anomaly detection head combining spectral + spatial views.

    Produces an anomaly classification (2-class) that can be trained jointly
    with the main EMGNN classification loss.
    """

    def __init__(self, nfeat: int, hidden: int, n_eigvecs: int = 32):
        super().__init__()
        self.spectral = SpectralAnomalyModule(n_eigvecs, hidden)
        self.spatial = SpatialAnomalyModule(nfeat, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                eigvecs: torch.Tensor) -> torch.Tensor:
        h_spec = self.spectral(eigvecs)
        h_spat = self.spatial(x, edge_index)
        h = torch.cat([h_spec, h_spat], dim=-1)
        return F.log_softmax(self.classifier(h), dim=1)


def precompute_laplacian_eigvecs(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 32,
) -> torch.Tensor:
    """
    Precompute the smallest-k eigenvectors of the symmetric normalised
    Laplacian. These are computed once and cached.

    Returns
    -------
    eigvecs : (num_nodes, k)
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    from torch_geometric.utils import get_laplacian

    L_index, L_weight = get_laplacian(
        edge_index, normalization='sym', num_nodes=num_nodes)
    L_sparse = sp.coo_matrix(
        (L_weight.cpu().numpy(), L_index.cpu().numpy()),
        shape=(num_nodes, num_nodes))
    L_sparse = L_sparse.tocsr()

    # Compute smallest-k eigenvalues/vectors (excluding trivial zero)
    k_actual = min(k, num_nodes - 2)
    eigenvalues, eigenvectors = eigsh(L_sparse, k=k_actual, which='SM')
    return torch.from_numpy(eigenvectors).float()
