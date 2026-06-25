# Cancer Driver Gene Prediction via Heterophily-Aware Graph Neural Networks

**NUS Capstone Project** — Final-year undergraduate thesis  
**Author:** Yixi Wang (c2h4wang@u.nus.edu)  
**Supervisor:** Department of Computer Science, School of Computing, NUS

Reference: [Chatzianastasis et al., *Bioinformatics* 2023](https://doi.org/10.1093/bioinformatics/btad643)

---

## Quick Status (June 2026)

| Item | Status |
|------|--------|
| M1 Benchmark reproduction | ✅ 6 networks × 3 backbones, 36+ runs |
| M2 Ablation + Optuna | ✅ BatchNorm harmful (−4.2%), 50 trials |
| M3 Multi-network extension | ✅ 6-net AUPR=0.8067, gain decomposition |
| M4 Interpretability (IG + GSEA) | ✅ 28 Hallmark pathways (FDR<0.05) |
| M5 Advanced technique ablation | ✅ 5/9 evaluated; P7 heterophily best (+2.8%) |
| External data modules (P5, P6, P8) | Optional future work |
| GNNExplainer (P10) | Optional future work |
| Paper (IMRAD) | Local-only artifact, not versioned in GitHub |

**Best scoped result:** Heterophily-aware gating + 6-network EMGNNImproved: **AUPR=0.8240±0.0044, AUROC=0.9194** over 3 legacy seeds.

---

## Project Overview

Identifying cancer driver genes from the vast background of passenger mutations is a central challenge in computational oncology. This project extends the Explainable Multilayer Graph Neural Network (EMGNN) framework to predict cancer driver genes by integrating six protein–protein interaction (PPI) networks with 64-dimensional pan-cancer multi-omics features.

**Core technical insight:** Prior work suggests that PPI-based cancer driver prediction is affected by heterophily, where neighbours of drivers are often non-drivers. The completed ablation results show that a learned gate fusing low-pass and high-pass filtered signals improves the six-network baseline without architectural redesign.

---

## Five Methodologies

| # | Methodology | Key Result | Status |
|---|------------|------------|--------|
| M1 | Benchmark reproduction | GCN mean AUPR=0.743 (18 runs), relative ranking GCN>GIN>GAT reproduced | ✅ |
| M2 | Ablation + Optuna | **BatchNorm harmful in full-batch GNN** (−4.2% AUPR); Optuna best=0.8023 (seed-dependent) | ✅ |
| M3 | Multi-network extension | 6-net AUPR=**0.8067** (+5.9%); gain: data +5.4%, architecture +1.0% | ✅ |
| M4 | Interpretability | IG: methylation dominates; GSEA: 28 Hallmark significant (EMT FDR=1.6×10⁻³³) | ✅ |
| M5 | Advanced technique ablation | 9 techniques implemented; **P7 heterophily best** (+2.8%, p<0.01) | ✅ 5/9 |

---

## M5 Ablation Results (RTX 5090, June 2026)

| Technique | Mean AUPR ↑ | ± std | Δ AUPR | Verdict |
|-----------|------------|-------|--------|---------|
| Baseline (6-net, 5 seeds) | 0.8019 | 0.0050 | — | Robust baseline |
| **Heterophily-aware gating (P7)** | **0.8240** | **0.0044** | **+2.8%** | 🏆 Best (p<0.01) |
| Cross-Network Attention (P4) | 0.8142 | 0.0053 | +1.5% | 🥈 Stable gain |
| DropEdge p=0.1 (P9) | 0.8064 | 0.0061 | +0.6% | 🥉 Marginal |
| GraphMAE pretraining (P2) | 0.8002 | 0.0075 | −0.2% | Neutral |
| Focal Loss γ=2 (P0) | 0.7608 | 0.0045 | −5.1% | ❌ Harmful |

**Out-of-scope extensions:** Pathway hypergraph, HIPGNN, PINNACLE embeddings, and GNNExplainer are retained in the codebase as optional future work and are not part of the final required experiment scope.

---

## Repository Structure

```
NUS-Capstone/
├── benchmark/                  # Original EMGNN reference code (M1)
├── src/
│   ├── models/
│   │   ├── emgnn_improved.py   # ★ Core model (M2/M3/M5, 9 modular techniques)
│   │   ├── hypergnn.py         # Pathway hypergraph encoder (P5)
│   │   ├── hipgnn.py           # Anomaly detection head (P6)
│   │   └── baselines.py        # GCN/MLP baselines
│   ├── data/
│   │   ├── loader.py           # Multi-network HDF5 loader + sparse cache
│   │   ├── feature_engineering.py
│   │   ├── build_hypergraph.py # GMT → hypergraph incidence (P5)
│   │   └── pinnacle_embeddings.py
│   ├── training/
│   │   ├── trainer.py          # Trainer: LR sched, early stop, grad clip
│   │   ├── hparam_search.py    # Optuna Bayesian search (M2)
│   │   └── pretrain_graphmae.py
│   └── explainability/
│       ├── attribution.py      # Integrated Gradients (Captum)
│       └── gsea.py             # Enrichr ORA + preranked + Hallmark
│
├── experiments/
│   ├── run_improved.py         # ★ Main training script (40+ CLI args)
│   ├── run_benchmark.py        # Original EMGNN runner
│   ├── run_hparam_search.py    # Optuna driver
│   ├── run_attribution.py      # Feature attribution
│   ├── run_gsea.py             # Gene set enrichment
│   ├── run_gnn_explain.py      # GNNExplainer (P10, fixed)
│   ├── run_remaining.sh        # Serial experiment runner (used June 2026)
│   └── run_m5_ablation.sh      # M5 ablation dispatcher
│
├── results/
│   ├── m5_ablation_summary.csv # ★ Complete M5 results
│   └── experiment_summary.md   # Detailed M1-M4 results
│
├── configs/
└── requirements.txt
```

---

## Setup

### Environment

```bash
conda create -n cancer-gnn python=3.10
conda activate cancer-gnn
pip install torch torch_geometric captum optuna gseapy h5py scikit-learn pandas
```

For RTX 5090 (CUDA 12.8):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
```

### Data

Download from the EMOGI benchmark (Zenodo record 3707301). Six HDF5 files expected under `results/EMOGI_*/` containing adjacency matrix, 64-dim multi-omics features (MF, METH, GE, CNA × 16 cancer types), and binary labels.

---

## Quick Start

```bash
# Best configuration (6-network, no BatchNorm)
python experiments/run_improved.py --gcn 1 \
    --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \
    --norm_type none --use_residual True --use_net_weights True \
    --lr_scheduler cosine --label_smoothing 0.05

# Heterophily-aware (best single technique, +2.8%)
python experiments/run_improved.py --gcn 1 \
    --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \
    --norm_type none --use_residual True --use_net_weights True \
    --lr_scheduler cosine --label_smoothing 0.05 \
    --heterophily_aware 1

# Cross-network attention (+1.5%)
python experiments/run_improved.py ... --cross_network_attention 1

# M5 ablation batch (all 5 evaluated techniques × 3 seeds)
bash experiments/run_remaining.sh
```

### Interpretability

```bash
# Feature attribution
python experiments/run_attribution.py --model_dir results/my_models/<dir>

# Gene set enrichment
python experiments/run_gsea.py --model_dir results/my_models/<dir> --mode enrichr --top_n 200
```

### CPU-Only Reproducibility Checks

```bash
python -m unittest discover -s tests
python scripts/smoke_test.py
python scripts/analyze_m5_results.py
```

The smoke test and summary script are CPU-only; no GPU or EMOGI HDF5 files are
required for these checks.

---

## Key Findings

1. **Heterophily-aware gating is the most effective single improvement** (+2.8%, p<0.01). The result is consistent with prior evidence that PPI-based cancer driver prediction is affected by heterophily.

2. **Multi-network data trumps architectural complexity.** The +5.9% gain decomposes into +5.4% (data) +1.0% (architecture). Adding more PPI databases is more impactful than model changes.

3. **BatchNorm is harmful in full-batch graph learning** (−4.2%). In full-batch training, running statistics provide no regularisation benefit.

4. **Focal Loss degrades performance** (−5.1%). A plausible explanation is that label smoothing (ε=0.05) already provides calibration and the additional γ modulation over-penalises the minority class; this mechanism was not separately validated by a 2x2 interaction ablation.

5. **GraphMAE pretraining is neutral** (−0.2%). The 64-dimensional multi-omics features are already sufficiently informative for supervised learning.

---

## Documentation

- **Result provenance:** [`results/RESULT_PROVENANCE.md`](results/RESULT_PROVENANCE.md)
- **Experiment summary:** [`results/experiment_summary.md`](results/experiment_summary.md)
- **M5 statistical summary:** [`results/m5_statistical_summary.csv`](results/m5_statistical_summary.csv)

---

## Citation

```bibtex
@article{chatzianastasis2023emgnn,
  title   = {{Explainable Multilayer Graph Neural Network for Cancer Gene Prediction}},
  author  = {Chatzianastasis, Michail and Vazirgiannis, Michalis and Zhang, Zijun},
  journal = {Bioinformatics},
  volume  = {39},
  number  = {11},
  pages   = {btad643},
  year    = {2023},
  doi     = {10.1093/bioinformatics/btad643}
}
```
