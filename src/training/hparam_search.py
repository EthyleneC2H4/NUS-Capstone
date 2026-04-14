"""
Bayesian hyperparameter optimisation using Optuna.

Searches over the most impactful hyperparameters for EMGNN:
  lr, hidden, n_layers, dropout, weight_decay,
  use_residual, use_batchnorm, label_smoothing, lr_scheduler.

Usage
-----
    python experiments/run_hparam_search.py --dataset CPDB --n_trials 50
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def run_hparam_search(
    dataset_names: list[str],
    n_trials: int = 50,
    epochs_per_trial: int = 500,
    patience_per_trial: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """
    Run Optuna Bayesian search over EMGNN hyperparameters.

    Parameters
    ----------
    dataset_names : list of str
        Networks to train on; last entry = test set.
    n_trials : int
        Number of Optuna trials.
    epochs_per_trial : int
        Max training epochs per trial (shorter than final run to save time).
    patience_per_trial : int
        Early-stopping patience per trial.
    device : str
    study_name : str | None
        Optuna study name (enables resuming from storage).
    storage : str | None
        SQLite or PostgreSQL URL for persistent study.

    Returns
    -------
    dict with keys 'best_params', 'best_value', 'study'
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Install optuna: pip install optuna")

    from torch_geometric.data import DataLoader

    # Import here to avoid circular deps at module level
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from src.data.loader import load_multi_network_data
    from src.models.emgnn_improved import EMGNNImproved
    from src.training.trainer import Trainer

    # Pre-load data once (shared across trials)
    print(f"Loading data for networks: {dataset_names}")
    loader, info = load_multi_network_data(dataset_names)
    batch = info['batch']
    nfeat = batch.x.shape[1]

    # ── Objective ─────────────────────────────────────────────────────────────

    def objective(trial: 'optuna.Trial') -> float:
        lr            = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        hidden        = trial.suggest_categorical('hidden', [32, 64, 128, 256])
        n_layers      = trial.suggest_int('n_layers', 1, 5)
        dropout       = trial.suggest_float('dropout', 0.1, 0.7)
        weight_decay  = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        use_residual  = trial.suggest_categorical('use_residual',  [True, False])
        use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
        label_smooth  = trial.suggest_float('label_smoothing', 0.0, 0.2)
        lr_scheduler  = trial.suggest_categorical('lr_scheduler',
                                                   ['none', 'cosine', 'step'])

        # Build a minimal args namespace
        args = argparse.Namespace(
            gcn=True, gat=False, gin=False, sage=False,
            hidden=hidden, n_layers=n_layers, dropout=dropout,
            alpha=0.2, nb_heads=1,
            lr=lr, weight_decay=weight_decay,
            epochs=epochs_per_trial, patience=patience_per_trial,
            lr_scheduler=None if lr_scheduler == 'none' else lr_scheduler,
            cuda=device.startswith('cuda'),
        )

        model = EMGNNImproved(
            nfeat=nfeat,
            hidden_channels=hidden,
            n_layers=n_layers,
            nclass=2,
            meta_x=info['meta_x'],
            args=args,
            data=info['batch'],
            node2idx=info['node2idx'],
            use_residual=use_residual,
            use_batchnorm=use_batchnorm,
            label_smoothing=label_smooth,
        ).to(device)

        # Move tensors to device
        meta_y   = info['meta_y'].to(device)
        idx_tr   = info['idx_train'].to(device)
        idx_vl   = info['idx_val'].to(device)
        idx_te   = info['idx_test'].to(device)

        trainer = Trainer(
            model=model, args=args,
            idx_train=idx_tr, idx_val=idx_vl, idx_test=idx_te,
            meta_y=meta_y,
            number_of_input_nodes=info['number_of_input_nodes'],
            device=device,
        )

        history = trainer.fit(loader)
        best_val_aupr = max(history['val_aupr']) if history['val_aupr'] else 0.0
        return best_val_aupr

    # ── Create and run study ──────────────────────────────────────────────────

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=study_name,
        storage=storage,
        load_if_exists=(storage is not None),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n=== Hyperparameter Search Complete ===")
    print(f"Best validation AUPR : {study.best_value:.4f}")
    print(f"Best hyperparameters : {study.best_params}")

    return {
        'best_params': study.best_params,
        'best_value':  study.best_value,
        'study':       study,
    }
