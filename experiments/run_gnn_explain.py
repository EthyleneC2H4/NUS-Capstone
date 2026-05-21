"""
Edge- and feature-level explanations using GNNExplainer / PGExplainer.

Complements the existing Integrated-Gradients attribution (run_attribution.py)
by revealing *which PPI edges* (not just which features) drive each prediction.

Usage
-----
    python experiments/run_gnn_explain.py \
        --model_dir results/my_models/EMGNNImproved_GCN_CPDB_2026_04_15_10_22_58 \
        --top_k 20 --algorithm gnnexplainer
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmark'))

from torch_geometric.explain import Explainer, GNNExplainer
from src.models.emgnn_improved import EMGNNImproved


# ── Wrapper to adapt EMGNNImproved to PyG Explainer interface ────────────────

class _ExplainWrapper(torch.nn.Module):
    """
    PyG Explainer expects model(x, edge_index, **kwargs).
    EMGNNImproved expects model(x, edge_index, data, ...).
    This wrapper binds `data` at construction time.
    """

    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data

    def forward(self, x, edge_index, **kwargs):
        return self.model(x, edge_index, self.data, **kwargs)


# ── Main ─────────────────────────────────────────────────────────────────────

def explain_top_genes(model_dir: str, top_k: int = 20,
                      algorithm: str = 'gnnexplainer'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load artefacts
    with open(f'{model_dir}/args.pkl', 'rb') as f:
        args = pickle.load(f)
    with open(f'{model_dir}/batch.pkl', 'rb') as f:
        batch = pickle.load(f)
    with open(f'{model_dir}/node2idx.pkl', 'rb') as f:
        node2idx = pickle.load(f)
    with open(f'{model_dir}/meta_x.pkl', 'rb') as f:
        meta_x = pickle.load(f)
    with open(f'{model_dir}/meta_edge_index.pkl', 'rb') as f:
        meta_edge_index = pickle.load(f)

    # Rebuild model
    model = EMGNNImproved(
        nfeat=batch.x.shape[1],
        hidden_channels=args.hidden,
        n_layers=args.n_layers,
        nclass=2,
        meta_x=meta_x,
        args=args,
        data=batch,
        node2idx=node2idx,
        use_residual=args.use_residual,
        use_batchnorm=args.use_batchnorm,
        norm_type=getattr(args, 'norm_type', 'batch'),
        use_network_weights=getattr(args, 'use_net_weights', True),
    ).to(device)
    model.load_state_dict(
        torch.load(f'{model_dir}/model.pkl', map_location=device)
    )
    model.eval()

    batch = batch.to(device)
    n_input = batch.x.shape[0]

    # Forward pass to get predictions
    with torch.no_grad():
        out = model(batch.x.float(), batch.edge_index, batch)
    meta_probs = torch.exp(out[n_input:, 1])  # P(cancer)
    top_indices = meta_probs.argsort(descending=True)[:top_k]

    idx2name = {v: k[1] for k, v in node2idx.items()}

    # Build explainer
    wrapper = _ExplainWrapper(model, batch)
    algo = GNNExplainer(epochs=200, lr=0.01)
    explainer = Explainer(
        model=wrapper,
        algorithm=algo,
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    # Build full edge_index (input graph + meta graph)
    edge_all = torch.cat(
        [batch.edge_index, meta_edge_index.to(device)], dim=1
    )
    x_all = torch.cat([batch.x.float(), meta_x.to(device)], dim=0)

    results = []
    for rank, meta_idx in enumerate(top_indices):
        gene = idx2name.get(meta_idx.item(), 'Unknown')
        prob = meta_probs[meta_idx].item()
        global_idx = n_input + meta_idx.item()

        print(f"[{rank + 1}/{top_k}] {gene}  P(cancer)={prob:.4f} ...", end=' ')
        try:
            explanation = explainer(x_all, edge_all, index=global_idx, target=1)
            edge_mask = explanation.edge_mask
            top_edges = edge_mask.argsort(descending=True)[:20] if edge_mask is not None else None
            feat_imp = (explanation.node_mask[global_idx].cpu().numpy()
                        if explanation.node_mask is not None else None)

            results.append(dict(
                gene=gene, prob=prob, rank=rank + 1,
                top_edge_indices=(top_edges.cpu().numpy().tolist()
                                  if top_edges is not None else None),
                feature_importance=(feat_imp.tolist()
                                    if feat_imp is not None else None),
            ))
            print('OK')
        except Exception as e:
            print(f'SKIP ({e})')

    # Save
    out_pkl = f'{model_dir}/gnn_explanations.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)

    out_csv = f'{model_dir}/gnn_explanations.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rank', 'gene', 'prob'])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in ['rank', 'gene', 'prob']})

    print(f"\nSaved {len(results)} explanations → {out_pkl}")
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='GNNExplainer for EMGNNImproved')
    p.add_argument('--model_dir', required=True)
    p.add_argument('--top_k', type=int, default=20)
    p.add_argument('--algorithm', default='gnnexplainer',
                   choices=['gnnexplainer'])
    args = p.parse_args()
    explain_top_genes(args.model_dir, args.top_k, args.algorithm)
