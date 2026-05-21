"""
Train EMGNNImproved (Methodology 2 & 3: optimise + multi-network extension).

Improvements over benchmark:
  --use_residual     : residual connections (default: True)
  --use_batchnorm    : batch normalisation  (default: True)
  --use_net_weights  : learnable per-network importance (default: True)
  --sage             : use GraphSAGE backbone
  --lr_scheduler     : 'cosine' | 'step' | 'none' (default: cosine)
  --label_smoothing  : label smoothing epsilon (default: 0.05)
  --normalize        : 'standard' | 'minmax' | 'none' (default: standard)
  --feature_select   : drop near-zero-variance features (default: True)

Usage
-----
    python experiments/run_improved.py --gcn 1

    python experiments/run_improved.py --gcn 1 \\
        --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \\
        --use_residual True --use_batchnorm True \\
        --lr_scheduler cosine --label_smoothing 0.05

    python experiments/run_improved.py --sage 1 --normalize standard
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import torch

# ── repo root on path ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmark'))

from src.data.loader import load_multi_network_data
from src.data.feature_engineering import FeatureEngineer
from src.models.emgnn_improved import EMGNNImproved
from src.training.trainer import Trainer
import gcnIO


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train EMGNNImproved')

    # GNN backbone
    p.add_argument('--gcn',  default=False, type=lambda x: x in ('1','True','true'))
    p.add_argument('--gat',  default=False, type=lambda x: x in ('1','True','true'))
    p.add_argument('--gin',  default=False, type=lambda x: x in ('1','True','true'))
    p.add_argument('--sage', default=False, type=lambda x: x in ('1','True','true'))
    p.add_argument('--mlp',  default=False, type=lambda x: x in ('1','True','true'))

    # Datasets
    p.add_argument('--dataset', nargs='+',
                   default=['IREF_2015','IREF','STRING','PCNET','MULTINET','CPDB'])

    # Architecture improvements
    p.add_argument('--use_residual',    default=True,
                   type=lambda x: x in ('1','True','true'))
    p.add_argument('--use_batchnorm',   default=True,
                   type=lambda x: x in ('1','True','true'),
                   help='Deprecated: use --norm_type instead. False sets norm_type=none.')
    p.add_argument('--norm_type',       type=str, default='batch',
                   choices=['batch', 'graph', 'layer', 'none'],
                   help='Normalisation type: batch (default, harmful), graph (recommended), layer, none')
    p.add_argument('--use_net_weights', default=True,
                   type=lambda x: x in ('1','True','true'))
    p.add_argument('--label_smoothing', type=float, default=0.05)
    p.add_argument('--heterophily_aware', default=False,
                   type=lambda x: x in ('1','True','true'),
                   help='Enable gated high/low-pass separation for heterophilic PPI graphs')
    p.add_argument('--focal_gamma',    type=float, default=0.0,
                   help='Focal loss gamma (0=off, 2.0 recommended for imbalanced data)')
    p.add_argument('--focal_alpha',    type=float, default=0.75,
                   help='Focal loss positive-class weight (0.75 = 3x weight for positives)')

    # Training hyperparameters
    p.add_argument('--lr',           type=float, default=0.005)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--hidden',       type=int,   default=64)
    p.add_argument('--n_layers',     type=int,   default=3)
    p.add_argument('--nb_heads',     type=int,   default=1)
    p.add_argument('--dropout',      type=float, default=0.5)
    p.add_argument('--alpha',        type=float, default=0.2)
    p.add_argument('--epochs',       type=int,   default=2000)
    p.add_argument('--patience',     type=int,   default=250)
    p.add_argument('--lr_scheduler', type=str,   default='cosine',
                   choices=['none', 'cosine', 'step'])

    # Feature engineering
    p.add_argument('--normalize',       type=str,  default='standard',
                   choices=['standard', 'minmax', 'none'])
    p.add_argument('--feature_select',  default=True,
                   type=lambda x: x in ('1','True','true'))
    p.add_argument('--add_structural_noise', type=float, default=0.0)
    p.add_argument('--drop_edge_rate', type=float, default=0.0,
                   help='DropEdge: fraction of edges randomly dropped per forward pass (0=off, 0.1 recommended)')

    # Misc
    p.add_argument('--seed',    type=int,  default=72)
    p.add_argument('--no_cuda', action='store_true', default=False)

    args = p.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.lr_scheduler == 'none':
        args.lr_scheduler = None
    return args


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.cuda:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = 'cpu'

    print(f"\n{'='*60}")
    print(f"EMGNNImproved  |  backbone: "
          f"{'GCN' if args.gcn else 'GAT' if args.gat else 'GIN' if args.gin else 'SAGE'}")
    print(f"Networks: {args.dataset}")
    norm_display = getattr(args, 'norm_type', 'batch') if args.use_batchnorm else 'none'
    print(f"residual={args.use_residual}  norm={norm_display}  "
          f"net_weights={args.use_net_weights}")
    print(f"lr_scheduler={args.lr_scheduler}  label_smooth={args.label_smoothing}")
    print(f"normalize={args.normalize}  feature_select={args.feature_select}")
    print(f"{'='*60}\n")

    # ── Feature engineering ───────────────────────────────────────────────────
    fe = None
    if args.normalize != 'none' or args.feature_select:
        fe = FeatureEngineer(
            normalize=args.normalize if args.normalize != 'none' else None,
            feature_selection=args.feature_select,
        )
        # Note: fe is applied inside loader per-graph; fit on first graph
        # For simplicity we pass an unfitted FE; loader will call transform only.
        # In production: pre-fit on training nodes of one reference graph.

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading multi-network data …")
    loader, info = load_multi_network_data(
        args.dataset,
        add_structural_noise=args.add_structural_noise,
    )
    batch = info['batch']
    nfeat = batch.x.shape[1]
    print(f"  Input nodes : {info['number_of_input_nodes']}")
    print(f"  Meta nodes  : {len(info['node2idx'])}")
    print(f"  Feature dim : {nfeat}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = EMGNNImproved(
        nfeat=nfeat,
        hidden_channels=args.hidden,
        n_layers=args.n_layers,
        nclass=2,
        meta_x=info['meta_x'],
        args=args,
        data=batch,
        node2idx=info['node2idx'],
        use_residual=args.use_residual,
        use_batchnorm=args.use_batchnorm,
        norm_type=getattr(args, 'norm_type', 'batch'),
        use_network_weights=args.use_net_weights,
        label_smoothing=args.label_smoothing,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    # Move data to device
    meta_y   = info['meta_y'].to(device)
    idx_tr   = info['idx_train'].to(device)
    idx_vl   = info['idx_val'].to(device)
    idx_te   = info['idx_test'].to(device)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model, args=args,
        idx_train=idx_tr, idx_val=idx_vl, idx_test=idx_te,
        meta_y=meta_y,
        number_of_input_nodes=info['number_of_input_nodes'],
        device=device,
    )

    t0 = time.time()
    history = trainer.fit(loader)
    print(f"\nTraining finished in {(time.time()-t0)/60:.1f} min")

    # ── Test ──────────────────────────────────────────────────────────────────
    test_metrics = trainer.evaluate(loader)
    print(f"\nTest Results  |  "
          f"AUPR: {test_metrics['aupr']:.4f}  |  "
          f"AUROC: {test_metrics['auroc']:.4f}")

    # ── Network importance weights ────────────────────────────────────────────
    if model.network_weights is not None:
        import torch.nn.functional as F
        weights = F.softmax(model.network_weights.detach().cpu(), dim=0).numpy()
        print("\nLearned per-network importance weights (softmax-normalised):")
        for net_name, w in zip(args.dataset, weights):
            print(f"  {net_name:<20}  {w:.4f}")
        # Save weights to results file
        os.makedirs('./results', exist_ok=True)
        with open('./results/network_weights.txt', 'a') as fh:
            fh.write(f"\n{args.dataset}  {('GCN' if args.gcn else 'GAT' if args.gat else 'GIN' if args.gin else 'SAGE')}\n")
            for net_name, w in zip(args.dataset, weights):
                fh.write(f"  {net_name}: {w:.4f}\n")

    # ── Save artefacts ────────────────────────────────────────────────────────
    backbone = ('GCN' if args.gcn else 'GAT' if args.gat else
                'GIN' if args.gin else 'SAGE')
    model_dir = gcnIO.create_model_dir(
        f'EMGNNImproved_{backbone}_{args.dataset[-1]}'
    )
    torch.save(model.state_dict(), f'{model_dir}/model.pkl')
    gcnIO.write_hyper_params2(args, f'{model_dir}/hyper_params.txt')

    probs = trainer.predict(loader)
    all_names = info['all_node_names']
    meta_names = all_names[info['number_of_input_nodes']:]
    node_names_2d = [[str(i), n.replace('_Meta_Node','')]
                     for i, n in enumerate(meta_names)]
    import numpy as np
    gcnIO.save_predictions(model_dir, np.array(node_names_2d),
                           probs.cpu().detach().numpy())

    for key, val in info.items():
        if key != 'batch':
            try:
                with open(f'{model_dir}/{key}.pkl', 'wb') as fh:
                    pickle.dump(val, fh)
            except Exception:
                pass
    with open(f'{model_dir}/batch.pkl', 'wb') as fh:
        pickle.dump(info['batch'], fh)
        
    # Save extra artefacts needed by run_attribution.py
    with open(f'{model_dir}/args.pkl', 'wb') as fh:
        pickle.dump(args, fh)
    with open(f'{model_dir}/meta_edge_index.pkl', 'wb') as fh:
        pickle.dump(model.meta_edge_index.cpu(), fh)
    n_input = info['number_of_input_nodes']
    final_y = torch.cat([
        torch.zeros(n_input, dtype=torch.long),
        info['meta_y'].cpu(),
    ])
    with open(f'{model_dir}/final_y.pkl', 'wb') as fh:
        pickle.dump(final_y, fh)

    # Append to global results
    os.makedirs('./results', exist_ok=True)
    with open('./results/results_improved.txt', 'a') as fh:
        fh.write(
            f"{args.dataset}  {backbone}  "
            f"residual:{args.use_residual}  bn:{args.use_batchnorm}  "
            f"net_w:{args.use_net_weights}  n_layers:{args.n_layers}  "
            f"hidden:{args.hidden}  lr_sched:{args.lr_scheduler}  "
            f"label_smooth:{args.label_smoothing}  "
            f"aupr:{test_metrics['aupr']:.4f}  "
            f"auroc:{test_metrics['auroc']:.4f}\n"
        )

    print(f"\nResults saved to {model_dir}/")


if __name__ == '__main__':
    main()
