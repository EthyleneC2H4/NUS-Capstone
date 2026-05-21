"""
Data loading utilities that wrap the original gcnIO functions and add:
- Feature alignment across networks
- Multi-network batching
- Meta-node construction
- Integration with FeatureEngineer
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import AddRandomWalkPE

# Re-use benchmark I/O
import sys
sys.path.insert(0, str(Path(__file__).parents[2] / 'benchmark'))
from gcnIO import load_hdf_data, save_predictions  # noqa: E402


# ── Dataset path registry ──────────────────────────────────────────────────────

DATASET_PATHS: Dict[str, str] = {
    'CPDB':     './results/EMOGI_CPDB/CPDB_multiomics.h5',
    'IREF':     './results/EMOGI_IRefIndex/IREF_multiomics.h5',
    'IREF_2015':'./results/EMOGI_IRefIndex_2015/IREF_2015_multiomics.h5',
    'MULTINET': './results/EMOGI_Multinet/MULTINET_multiomics.h5',
    'PCNET':    './results/EMOGI_PCNet/PCNET_multiomics.h5',
    'STRING':   './results/EMOGI_STRINGdb/STRINGdb_multiomics.h5',
}

# Canonical feature order (64 = 4 omics × 16 cancer types)
FEATURES_ORDER: List[str] = (
    [f'MF: {c}' for c in
     ['UCEC','BLCA','THCA','KIRC','READ','LUAD','ESCA','LUSC',
      'BRCA','COAD','HNSC','KIRP','PRAD','LIHC','STAD','CESC']]
  + [f'METH: {c}' for c in
     ['UCEC','BLCA','THCA','KIRC','READ','LUAD','ESCA','LUSC',
      'BRCA','COAD','HNSC','KIRP','PRAD','LIHC','STAD','CESC']]
  + [f'GE: {c}' for c in
     ['UCEC','BLCA','THCA','KIRC','READ','LUAD','ESCA','LUSC',
      'BRCA','COAD','HNSC','KIRP','PRAD','LIHC','STAD','CESC']]
  + [f'CNA: {c}' for c in
     ['UCEC','BLCA','THCA','KIRC','READ','LUAD','ESCA','LUSC',
      'BRCA','COAD','HNSC','KIRP','PRAD','LIHC','STAD','CESC']]
)


# ── Main loading function ──────────────────────────────────────────────────────

def load_multi_network_data(
    dataset_names: List[str],
    add_structural_noise: float = 0.0,
    feature_engineer=None,
    pe_dim: int = 0,
) -> Tuple[DataLoader, dict]:
    """
    Load and prepare multi-network data for EMGNN training.

    Parameters
    ----------
    dataset_names : list of str
        PPI network identifiers. The **last** entry is used as the test set.
    add_structural_noise : float
        Probability of dropping each edge (0 = no noise).
    feature_engineer : FeatureEngineer | None
        Optional fitted FeatureEngineer to transform node features.

    Returns
    -------
    loader : DataLoader
        Single-batch DataLoader containing all graphs.
    info : dict
        Metadata dict with keys: node2idx, meta_x, meta_y, idx_train,
        idx_val, idx_test, number_of_input_nodes, all_node_names.
    """
    from torch_geometric.utils import add_self_loops, dropout_adj

    paths = [DATASET_PATHS[n] for n in dataset_names]

    data_list = []
    node2idx: Dict[tuple, int] = {}
    counter = 0

    train_nodes_list, val_nodes_list, test_nodes_list = [], [], []
    node_names_all = []
    y_list = []

    MAX_NODES = 100_000
    feat_dim = len(FEATURES_ORDER) + pe_dim  # 64 + pe_dim
    meta_y = torch.zeros(MAX_NODES, 1)
    meta_x_raw = torch.zeros(MAX_NODES, feat_dim)

    for path in paths:
        (adj, features, y_train, y_val, y_test,
         train_mask, val_mask, test_mask,
         node_names, feature_names) = load_hdf_data(path, feature_name='features')

        # ── Align features to canonical order ──────────────────────────────
        feat_names_dec = [f.decode('utf-8') if isinstance(f, bytes) else f
                          for f in feature_names]
        feat_idx = [feat_names_dec.index(fn) for fn in FEATURES_ORDER]
        features = features[:, feat_idx]

        # ── Optional feature engineering ───────────────────────────────────
        if feature_engineer is not None:
            features = feature_engineer.transform(features)

        # ── Build node2idx ─────────────────────────────────────────────────
        for node in node_names:
            key = tuple(node)
            if key not in node2idx:
                node2idx[key] = counter
                counter += 1

        y_train = y_train.astype(int)
        y_val   = y_val.astype(int)   if y_val is not None else np.zeros_like(y_train)
        y_test  = y_test.astype(int)

        y = (torch.tensor(y_train)
             .add(torch.tensor(y_test))
             .add(torch.tensor(y_val)))
        y_list.append(y)

        for i, label in enumerate(y):
            idx = node2idx[tuple(node_names[i])]
            meta_x_raw[idx] = data.x[i]  # includes PE if pe_dim > 0
            if meta_y[idx] == 0:
                meta_y[idx] = label.float()

        # ── Build edge index (sparse-safe, avoids huge dense tensor) ──────
        import scipy.sparse as sp
        if sp.issparse(adj):
            rows, cols = adj.nonzero()
            edge_index = torch.tensor(
                np.vstack([rows, cols]), dtype=torch.long
            ).contiguous()
        else:
            adj_tensor = torch.FloatTensor(np.array(adj))
            edge_index = (adj_tensor > 0).nonzero().t().contiguous()
        edge_index, _ = add_self_loops(edge_index)

        if add_structural_noise > 0:
            edge_index = dropout_adj(
                edge_index, p=add_structural_noise, force_undirected=True
            )[0]

        idx_train = torch.LongTensor([i for i, x in enumerate(train_mask) if x])
        idx_val   = torch.LongTensor([i for i, x in enumerate(val_mask)   if x])
        idx_test  = torch.LongTensor([i for i, x in enumerate(test_mask)  if x])

        train_nodes_list.append(node_names[idx_train.numpy()])
        val_nodes_list.append(node_names[idx_val.numpy()])
        test_nodes_list.append(node_names[idx_test.numpy()])
        node_names_all.append(node_names)

        features_t = torch.FloatTensor(features)
        data = Data(x=features_t, edge_index=edge_index, y=y,
                    node_names=node_names)

        # ── Optional Random-Walk positional encoding ──────────────────────
        if pe_dim > 0:
            rw = AddRandomWalkPE(walk_length=pe_dim, attr_name='rw_pe')
            data = rw(data)
            data.x = torch.cat([data.x, data.rw_pe], dim=-1)  # (N, 64+pe_dim)

        data_list.append(data)

    n_unique = len(node2idx)
    meta_x = meta_x_raw[:n_unique]
    meta_y = torch.tensor(meta_y[:n_unique]).type(torch.LongTensor).squeeze()

    # ── Build global train / val / test sets on meta-nodes ─────────────────
    train_set = {tuple(n) for lst in train_nodes_list for n in lst}
    val_set   = {tuple(n) for lst in val_nodes_list   for n in lst}
    test_set  = {tuple(n) for n in test_nodes_list[-1]}   # last graph only

    # Take first 10 % of train as val (mirror benchmark)
    val_set_sub = set(itertools.islice(train_set, int(len(train_set) * 0.1)))
    val_set = val_set_sub

    train_set -= test_set
    val_set   -= test_set
    train_set -= val_set

    idx_train_meta = torch.tensor([node2idx[n] for n in train_set])
    idx_val_meta   = torch.tensor([node2idx[n] for n in val_set])
    idx_test_meta  = torch.tensor([node2idx[n] for n in test_set])

    # ── Batch all graphs ───────────────────────────────────────────────────
    loader = DataLoader(data_list, batch_size=len(data_list))
    batch  = next(iter(loader))

    # Build all_node_names (input nodes + meta nodes)
    concat_names = np.concatenate(node_names_all, axis=0)
    meta_node_names = np.empty(n_unique, dtype=object)
    for k, v in node2idx.items():
        meta_node_names[v] = k[1] + '_Meta_Node'
    all_node_names = np.concatenate([
        concat_names[:, 1],  # gene symbol column
        meta_node_names,
    ])

    # ── Optional: concatenate PINNACLE pretrained embeddings to meta_x ──
    # (handled at model level via meta_linear input dimension)

    info = dict(
        node2idx=node2idx,
        meta_x=meta_x,
        meta_y=meta_y,
        idx_train=idx_train_meta,
        idx_val=idx_val_meta,
        idx_test=idx_test_meta,
        number_of_input_nodes=batch.x.shape[0],
        all_node_names=all_node_names,
        batch=batch,
        dataset_names=dataset_names,
    )
    return loader, info
