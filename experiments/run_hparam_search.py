"""
Hyperparameter search for EMGNNImproved (Methodology 2: optimise model).

Uses Optuna's TPE sampler with median pruner.  Results are printed and
also written to ./results/hparam_search_results.csv.

Usage
-----
    python experiments/run_hparam_search.py \\
        --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \\
        --n_trials 50 --epochs_per_trial 500
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.training.hparam_search import run_hparam_search


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', nargs='+',
                   default=['IREF_2015','IREF','STRING','PCNET','MULTINET','CPDB'])
    p.add_argument('--n_trials',          type=int, default=50)
    p.add_argument('--epochs_per_trial',  type=int, default=500)
    p.add_argument('--patience_per_trial',type=int, default=100)
    p.add_argument('--study_name',        type=str, default=None)
    p.add_argument('--storage',           type=str, default=None,
                   help='SQLite URL for persistent study, e.g. sqlite:///study.db')
    args = p.parse_args()

    result = run_hparam_search(
        dataset_names=args.dataset,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        patience_per_trial=args.patience_per_trial,
        study_name=args.study_name,
        storage=args.storage,
    )

    # Save best params
    import csv, os
    os.makedirs('./results', exist_ok=True)
    with open('./results/hparam_search_results.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(result['best_params'].keys())
                           + ['best_aupr'])
        w.writeheader()
        row = dict(result['best_params'])
        row['best_aupr'] = result['best_value']
        w.writerow(row)
    print(f"\nBest params written to ./results/hparam_search_results.csv")


if __name__ == '__main__':
    main()
