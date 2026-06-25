# Reproducibility Guide

This guide describes CPU-safe checks, the current evidence trail, and optional
future training runs. It does not require GPU unless explicitly stated.

## Environment

```bash
conda env create -f environment.yml
conda activate cancer-gnn
```

or install the pinned pip dependencies:

```bash
pip install -r requirements.txt
```

## CPU Checks

Run all lightweight checks:

```bash
python -m unittest discover -s tests
python scripts/smoke_test.py
python scripts/analyze_m5_results.py
```

The smoke test uses synthetic graphs and does not require EMOGI HDF5 files.

## Existing Results Summary

Generate a statistical summary from the existing M5 CSV:

```bash
python scripts/analyze_m5_results.py \
  --input results/m5_ablation_summary.csv \
  --output results/m5_statistical_summary.csv
```

These are legacy statistics because the original runs predate saved split
manifests.

## Optional Heterophily Analysis

After copying/downloading the EMOGI HDF5 files to `results/EMOGI_*`, run:

```bash
python scripts/compute_network_homophily.py \
  --output results/network_homophily_summary.csv
```

This CPU-only script computes overall and class-specific edge homophily for each
PPI network.

## Optional Future Fixed-Split Training

Any future training should use a shared split seed:

```bash
python experiments/run_improved.py --gcn 1 \
  --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \
  --norm_type none --use_residual True --use_net_weights True \
  --lr_scheduler cosine --label_smoothing 0.05 \
  --seed 72 --split_seed 72
```

Each model directory will contain:

- `split_manifest.json`
- `experiment_metadata.json`
- `args.pkl`
- `predictions.tsv`
- `hyper_params.txt`

To reuse an exact split:

```bash
python experiments/run_improved.py --gcn 1 \
  --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \
  --split_file results/my_models/<model_dir>/split_manifest.json \
  --seed 1
```

## Evidence Standard

A paper-ready result must record:

- command and parsed arguments;
- Git commit and dirty-state flag;
- package/CUDA/Python versions;
- model seed and split seed;
- gene-level train/validation/test split;
- raw metrics and prediction file.
