"""
Build hypergraph from gene set collections (GMT format).

A hypergraph represents functional annotations (GO terms, KEGG pathways)
as hyperedges connecting all member genes. This provides pathway-level
structural priors that complement the PPI topology.

Data source: MSigDB GMT files (e.g., h.all.v2024.1.Hs.symbols.gmt)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def load_gmt(gmt_path: str) -> Dict[str, List[str]]:
    """Parse a GMT file (MSigDB standard format).

    Returns dict {set_name: [gene_symbol, ...]}.
    """
    gene_sets = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            set_name = parts[0]
            genes = parts[2:]  # skip description field
            gene_sets[set_name] = genes
    return gene_sets


def build_hypergraph_incidence(
    gene_sets: Dict[str, List[str]],
    node2idx: dict,
    min_members: int = 5,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Build hypergraph incidence matrix H from gene sets.

    Parameters
    ----------
    gene_sets : dict  {set_name: [gene_symbol, ...]}
    node2idx : dict   {(db_id, gene_symbol): meta_node_idx}
    min_members : int
        Minimum number of matched genes for a hyperedge to be included.

    Returns
    -------
    H : torch.FloatTensor (n_meta_nodes, n_hyperedges)
        Incidence matrix: H[i,j] = 1 if gene i belongs to gene set j.
    hyperedge_names : list of str
        Names of the retained hyperedges.
    """
    # Build gene_symbol → meta_idx mapping
    symbol2idx: Dict[str, int] = {}
    for (db_id, symbol), idx in node2idx.items():
        symbol2idx[symbol] = idx

    n_nodes = len(node2idx)
    hyperedge_names = []
    cols = []

    for set_name, genes in gene_sets.items():
        member_indices = [symbol2idx[g] for g in genes if g in symbol2idx]
        if len(member_indices) >= min_members:
            hyperedge_names.append(set_name)
            cols.append(member_indices)

    n_edges = len(cols)
    H = torch.zeros(n_nodes, n_edges)
    for j, members in enumerate(cols):
        H[members, j] = 1.0

    print(f"  Hypergraph: {n_edges} hyperedges from {len(gene_sets)} gene sets "
          f"({n_nodes} nodes, min_members={min_members})")
    return H, hyperedge_names
