"""
Run the original EMGNN benchmark (Methodology 1: reproduce benchmark).

This script is a thin wrapper around benchmark/train.py that calls it
with the default settings reported in the paper (Table 1).

Usage
-----
    # Reproduce Table 1 – test on CPDB (default)
    python experiments/run_benchmark.py --gcn 1

    # Test on STRING
    python experiments/run_benchmark.py --gcn 1 --dataset IREF_2015 IREF PCNET MULTINET CPDB STRING

    # Run GAT variant
    python experiments/run_benchmark.py --gat 1

    # Run MLP baseline
    python experiments/run_benchmark.py --mlp 1

    # Reproducible run (fixed random seed)
    python experiments/run_benchmark.py --gcn 1 --seed 42
"""

import subprocess
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parents[1] / 'benchmark'


def main():
    # Forward all arguments to the original train.py.
    # benchmark/train.py already accepts --seed (default 72); passing it here
    # makes reproducibility explicit at the experiments/ layer.
    cmd = [sys.executable, str(BENCHMARK_DIR / 'train.py')] + sys.argv[1:]
    print(f"Running benchmark: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(BENCHMARK_DIR))


if __name__ == '__main__':
    main()
