"""
Attribution analysis for EMGNNImproved (Methodology 4: interpretability).

Computes Integrated-Gradients edge and node-feature attributions for
cancer / non-cancer / top-predicted genes and saves:
  - edge_attribution_{idx}.pkl
  - feature_attribution_{idx}.pkl
  - feature_importance_aggregated.csv
  - feature_importance.pdf

Usage
-----
    python experiments/run_attribution.py \\
        --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \\
        --gene_label cancer \\
        --edge_explain --node_explain
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / 'benchmark'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dir', required=True)
    p.add_argument('--gene_label', default='cancer',
                   choices=['cancer', 'non_cancer', 'top_predicted'])
    p.add_argument('--edge_explain', action='store_true')
    p.add_argument('--node_explain', action='store_true', default=True)
    p.add_argument('--max_genes', type=int, default=50,
                   help='Max number of genes to explain (for speed)')
    args = p.parse_args()

    out_dir = os.path.join(args.model_dir, 'attribution')
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load saved artefacts ──────────────────────────────────────────────────
    def _load(name):
        with open(os.path.join(args.model_dir, name), 'rb') as fh:
            return pickle.load(fh)

    batch            = _load('batch.pkl')
    args_model       = _load('args.pkl')
    meta_x           = _load('meta_x.pkl')
    meta_edge_index  = _load('meta_edge_index.pkl')
    node2idx         = _load('node2idx.pkl')
    all_node_names   = _load('all_node_names.pkl')
    final_y          = _load('final_y.pkl')
    idx_test         = _load('idx_test.pkl')

    number_of_input_nodes = batch.x.shape[0]

    if hasattr(final_y, 'cuda'):
        final_y = final_y.squeeze()

    # ── Load model ────────────────────────────────────────────────────────────
    # Try EMGNNImproved first, fall back to benchmark EMGNN
    try:
        from src.models.emgnn_improved import EMGNNImproved
        model = EMGNNImproved(
            nfeat=batch.x.shape[1],
            hidden_channels=args_model.hidden,
            n_layers=args_model.n_layers,
            nclass=2,
            meta_x=meta_x,
            args=args_model,
            data=batch,
            node2idx=node2idx,
        ).to(device)
        model_type = 'EMGNNImproved'
    except Exception:
        from model import EMGNN
        model = EMGNN(
            nfeat=batch.x.shape[1],
            hidden_channels=args_model.hidden,
            n_layers=args_model.n_layers,
            nclass=2,
            meta_x=meta_x,
            args=args_model,
            data=batch,
            node2idx=node2idx,
        ).to(device)
        model_type = 'EMGNN'

    model.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'model.pkl'),
                   map_location=device)
    )
    model.eval()
    print(f"Loaded {model_type} from {args.model_dir}")

    batch = batch.to(device)
    meta_x = meta_x.to(device)
    meta_edge_index = meta_edge_index.to(device)
    final_y = final_y.to(device)

    # ── Select genes to explain ───────────────────────────────────────────────
    meta_labels = final_y[number_of_input_nodes:]
    if args.gene_label == 'cancer':
        indices = [i for i, l in enumerate(meta_labels) if l == 1]
    elif args.gene_label == 'non_cancer':
        indices = [i for i, l in enumerate(meta_labels) if l == 0]
    else:  # top_predicted
        preds_path = os.path.join(args.model_dir, 'predictions.tsv')
        df_pred = pd.read_csv(preds_path, sep='\t')
        df_pred = df_pred.sort_values('Prob_pos', ascending=False).head(100)
        name_to_idx = {n.replace('_Meta_Node', ''): i
                       for i, n in enumerate(all_node_names[number_of_input_nodes:])}
        indices = [name_to_idx[n] for n in df_pred['Name']
                   if n in name_to_idx]

    indices = indices[:args.max_genes]
    print(f"Explaining {len(indices)} {args.gene_label} genes …")

    # ── Attribution ───────────────────────────────────────────────────────────
    from src.explainability.attribution import AttributionAnalyzer

    analyzer = AttributionAnalyzer(
        model=model, batch=batch,
        meta_x=meta_x, meta_edge_index=meta_edge_index,
        number_of_input_nodes=number_of_input_nodes,
        all_node_names=all_node_names,
        final_y=final_y, device=device,
    )

    if args.node_explain:
        print("  Computing node-feature attributions …")
        feat_df = analyzer.aggregate_feature_importance(indices, mode='mean')
        feat_csv = os.path.join(out_dir, f'feature_importance_{args.gene_label}.csv')
        feat_df.to_csv(feat_csv, index=False)
        print(f"  Saved → {feat_csv}")
        print(f"  Top-5 features:\n{feat_df.head().to_string(index=False)}")

        fig = analyzer.plot_feature_importance(
            feat_df,
            title=f'Feature Importance: {args.gene_label} genes',
            save_path=os.path.join(out_dir, f'feature_importance_{args.gene_label}.pdf'),
        )

    if args.edge_explain:
        print("  Computing edge attributions …")
        for idx in indices[:10]:    # limit to 10 for speed
            _, normed = analyzer.edge_attribution(idx)
            analyzer.save(normed,
                          os.path.join(out_dir,
                                       f'edge_attr_{args.gene_label}_{idx}.pkl'))
        print(f"  Edge attributions saved to {out_dir}/")

    print(f"\nDone. Attribution outputs in: {out_dir}")


if __name__ == '__main__':
    main()
