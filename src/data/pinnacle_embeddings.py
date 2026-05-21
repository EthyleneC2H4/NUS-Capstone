"""
Load PINNACLE pretrained protein embeddings and map to project genes.

PINNACLE (Nature Methods 2024) provides 128-dim embeddings for ~19K proteins
learned from multi-tissue PPI networks. These embeddings capture protein
structure/function priors complementary to the omics features (MF/METH/GE/CNA).

Source: https://huggingface.co/datasets/mli/pinnacle-protein-embeddings
Expected format: .npz with 'genes' (array of gene symbols) and 'embeddings'
(n_genes × 128 float array).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_pinnacle_embeddings(
    embedding_path: str,
    node2idx: dict,
    embedding_dim: int = 128,
) -> torch.Tensor:
    """
    Load PINNACLE embeddings and map to meta-node indices.

    Parameters
    ----------
    embedding_path : str
        Path to PINNACLE .npz file with 'genes' and 'embeddings' keys.
    node2idx : dict
        {(db_id, gene_symbol): meta_node_idx}
    embedding_dim : int
        Embedding dimension (128 for PINNACLE).

    Returns
    -------
    embeddings : torch.FloatTensor (n_meta_nodes, embedding_dim)
        Unmatched genes are filled with zero vectors.
    """
    raw = np.load(embedding_path, allow_pickle=True)
    gene2embed = {
        gene: embed
        for gene, embed in zip(raw['genes'], raw['embeddings'])
    }

    n_meta = len(node2idx)
    embeddings = torch.zeros(n_meta, embedding_dim)

    matched = 0
    for (db_id, symbol), idx in node2idx.items():
        if symbol in gene2embed:
            embeddings[idx] = torch.from_numpy(
                gene2embed[symbol].astype(np.float32))
            matched += 1

    print(f"  PINNACLE: matched {matched}/{n_meta} genes "
          f"({matched / n_meta * 100:.1f}%)")
    return embeddings
