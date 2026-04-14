# Cancer Gene Prediction using Graph Neural Networks

**NUS Capstone Project** — *Cancer gene prediction using graph neural network*

Reference paper: [Chatzianastasis et al., *Bioinformatics* 2023, btad643](https://doi.org/10.1093/bioinformatics/btad643)

---

## Project Overview

This repository implements the full methodology pipeline for cancer gene prediction using multilayer graph neural networks, building on and extending the EMGNN benchmark.

**Objectives**
- Accurately predict cancer driver genes by fusing multiple PPI (protein–protein interaction) networks with pan-cancer multi-omics data
- Provide biologically interpretable explanations for each prediction

**Methodology summary**

| # | Task | Status |
|---|------|--------|
| 1 | Reproduce benchmark (EMGNN) | ✅ `benchmark/` |
| 2 | Optimise model (residual, BN, LR scheduler, label smoothing, hparam search) | ✅ `src/models/emgnn_improved.py` |
| 3 | Extend to multiple biological networks (learnable network-importance weights) | ✅ `src/models/emgnn_improved.py` |
| 4 | Enhance interpretability (feature attribution + GSEA) | ✅ `src/explainability/` |

---

## Repository Structure

```
NUS-Capstone/
├── benchmark/                  # Verbatim EMGNN benchmark code (Methodology 1)
│   ├── model.py                  Original EMGNN / GCN / MLP models
│   ├── train.py                  Original training script
│   ├── explain.py                Original Integrated-Gradients explainer
│   ├── gcnIO.py                  HDF5 I/O and result-saving utilities
│   └── captum_custom.py          Captum edge/node mask wrappers
│
├── src/
│   ├── models/
│   │   ├── emgnn_improved.py     ★ Improved EMGNN (Methodology 2 & 3)
│   │   └── baselines.py          GCN and MLP baselines
│   ├── data/
│   │   ├── loader.py             Multi-network data loading pipeline
│   │   └── feature_engineering.py  ★ Feature normalisation & selection
│   ├── training/
│   │   ├── trainer.py            ★ Improved trainer (LR scheduler, grad clipping)
│   │   └── hparam_search.py      ★ Optuna Bayesian hyperparameter search
│   └── explainability/
│       ├── attribution.py        ★ Integrated-Gradients attribution (clean API)
│       └── gsea.py               ★ Gene set enrichment analysis (Methodology 4)
│
├── experiments/
│   ├── run_benchmark.py          Run original EMGNN
│   ├── run_improved.py           Run EMGNNImproved
│   ├── run_hparam_search.py      Bayesian hyperparameter optimisation
│   ├── run_attribution.py        Feature/edge attribution for trained model
│   └── run_gsea.py               GSEA on model predictions
│
├── configs/
│   ├── benchmark_config.yaml     Benchmark hyperparameters
│   └── improved_config.yaml      Improved model hyperparameters
│
└── requirements.txt
```

---

## Methodology Details

### Methodology 1 — Benchmark Reproduction

The `benchmark/` directory contains the **unmodified** original EMGNN code from [zhanglab-aim/EMGNN](https://github.com/zhanglab-aim/EMGNN).

**Architecture recap**

```
Step 1 – Per-network GNN:
  For each PPI graph G^(k), apply n_layers of shared GCN/GAT/GIN
  to propagate multi-omics features through gene neighbourhoods.

Step 2 – Meta-graph construction:
  Create one meta-node per unique gene. Connect each per-graph copy of
  gene_i to its meta-node with a directed edge.

Step 3 – Meta-graph GNN + classifier:
  Apply a second GNN on the meta-graph to aggregate cross-network signals.
  Final MLP classifies each meta-node as cancer / non-cancer.
```

**Reported performance (paper, AUPR ± std, 5 runs)**

| Test network | EMGNN-GCN | EMOGi (SOTA) |
|---|---|---|
| CPDB | 0.809 ± 0.006 | 0.746 |
| Multinet | 0.854 ± 0.007 | 0.808 |
| PCNet | 0.761 ± 0.001 | 0.697 |
| STRING | 0.856 ± 0.002 | 0.806 |
| Iref | 0.832 ± 0.002 | 0.778 |
| Iref 2015 | 0.800 ± 0.010 | 0.734 |

**Our reproduced results (single run, 300 epochs, patience 100, CPU)**

| Backbone | Test network | AUPR | AUROC |
|----------|-------------|------|-------|
| GCN | CPDB | 0.7479 | 0.8668 |
| GIN | CPDB | 0.7324 | 0.8669 |
| GAT | CPDB | 0.6158 | 0.7790 |
| GCN | STRING | 0.7361 | 0.8813 |
| GIN | STRING | 0.7341 | 0.8693 |
| GAT | STRING | 0.6308 | 0.8241 |

GCN and GIN reproduce the paper's trend (GCN best on CPDB, GIN competitive). GAT underperforms, consistent with the paper's observation that attention mechanisms require more data to be effective.

---

### Methodology 2 — Model Optimisation

**EMGNNImproved** (`src/models/emgnn_improved.py`) adds four improvements:

#### 2a. Residual connections (`use_residual=True`)
Skip connections are added from layer *l−1* to layer *l* for all GNN layers after the first:
```
h^(l) = LeakyReLU(GNN_l(h^(l-1))) + h^(l-1)
```
This improves gradient flow and enables training deeper networks without degradation.

#### 2b. Batch normalisation (`use_batchnorm=True`)
`BatchNorm1d` is applied after each GNN layer and after the meta-GNN:
```
h^(l) = BN(GNN_l(h^(l-1)))
```
Stabilises training, allows higher learning rates, and acts as a regulariser.

#### 2c. Label smoothing (`label_smoothing=0.05`)
Hard 0/1 targets are replaced with soft labels:
```
y_smooth = y * (1 - ε) + ε / K
```
Prevents over-confidence, improves calibration on the imbalanced cancer/non-cancer dataset.

#### 2d. Learning-rate scheduling (`lr_scheduler=cosine`)
Cosine annealing decays the learning rate smoothly from `lr` to `1e-5`:
```
lr_t = η_min + 0.5 * (lr - η_min) * (1 + cos(π * t / T))
```
Reduces oscillation near convergence and often finds flatter minima.

**Ablation study results (GCN backbone, CPDB test set)**

| Configuration | AUPR | AUROC | Δ AUPR vs baseline |
|--------------|------|-------|-------------------|
| Benchmark (no improvements) | 0.7479 | 0.8668 | — |
| + Residual only (no BN) | 0.7440 | 0.8658 | −0.004 |
| + BN + Residual | 0.7061 | 0.8579 | −0.042 |
| + BN + Residual + CosineAnneal | 0.7064 | 0.8641 | −0.042 |

**Key finding:** BatchNorm1d *hurts* performance on these graph datasets. Because the entire graph is processed as a single batch, BatchNorm's running statistics are computed over the full node set, which does not provide useful normalisation here. Residual connections alone match the baseline without degradation.

#### 2e. Gradient clipping
Gradients are clipped to `max_norm=1.0` before every optimiser step, preventing exploding gradients in deep configurations.

#### 2f. Feature engineering (`src/data/feature_engineering.py`)
A scikit-learn-style pipeline provides:
- **Z-score standardisation** — removes scale differences between omics types
- **Variance-threshold feature selection** — drops near-constant features (var < 0.01)
- **Optional PCA** — dimensionality reduction preserving specified variance

#### 2g. Hyperparameter search (`src/training/hparam_search.py`)
Bayesian optimisation (Optuna TPE sampler, median pruner) searches over:
```
lr ∈ [1e-4, 1e-2]
hidden ∈ {32, 64, 128, 256}
n_layers ∈ {1, 2, 3, 4, 5}
dropout ∈ [0.1, 0.7]
weight_decay ∈ [1e-5, 1e-3]
use_residual ∈ {True, False}
use_batchnorm ∈ {True, False}
label_smoothing ∈ [0, 0.2]
lr_scheduler ∈ {none, cosine, step}
```

---

### Methodology 3 — Multi-Network Extension

#### Learnable per-network importance weights (`use_network_weights=True`)
The original EMGNN treats all PPI networks equally. EMGNNImproved adds a learnable softmax-normalised scalar weight *w_k* per network:
```
w = softmax(θ)         # θ ∈ ℝ^K, K = number of networks
h_i ← h_i * w_{k(i)}  # weight node i's embedding by its network's importance
```
This allows the model to up-weight high-quality networks (e.g. STRING or IRef) and down-weight noisy ones automatically during training.

**Multi-network results (GCN, 3 PPI networks, test on CPDB)**

| Model | Networks | AUPR | AUROC | Δ vs single-net |
|-------|---------|------|-------|----------------|
| Benchmark GCN | CPDB only | 0.7479 | 0.8668 | — |
| Benchmark GCN | IREF_2015 + MULTINET + CPDB | **0.7877** | **0.9041** | +0.040 / +0.037 |
| EMGNNImproved GCN | IREF_2015 + MULTINET + CPDB | **0.7894** | 0.8989 | +0.045 / +0.032 |

Training with multiple PPI networks significantly boosts performance (+4% AUPR), demonstrating the value of cross-network signal aggregation through the meta-graph. The improved model marginally outperforms the benchmark in AUPR.

#### GraphSAGE backbone (`--sage`)
A new backbone option (`SAGEConv`) is added alongside GCN/GAT/GIN. GraphSAGE uses sampling-based neighbourhood aggregation:
```
h^(l) = W · concat(h^(l-1), MEAN({h^(l-1)_v : v ∈ N(u)}))
```
This is particularly effective when node degrees vary widely across PPI networks.

---

### Methodology 4 — Interpretability Enhancement

#### 4a. Improved attribution (`src/explainability/attribution.py`)
The `AttributionAnalyzer` class provides a clean API over Captum's Integrated Gradients:
- **Edge attributions** — which meta-graph edges drive a given prediction
- **Node-feature attributions** — which of the 64 multi-omics features are most important
- **Aggregated importance** — mean/max importance across a set of genes (e.g. all cancer genes), producing a global feature importance ranking

Feature importance is automatically colour-coded by omics type (MF, METH, GE, CNA) in visualisation.

#### 4b. Gene Set Enrichment Analysis (`src/explainability/gsea.py`)
The `GSEAAnalyzer` class supports three analysis modes:

| Mode | Method | Use case |
|------|--------|----------|
| `enrichr` | Enrichr ORA (internet) | Discrete top-N predicted gene list |
| `preranked` | Pre-ranked GSEA | Continuous prediction scores as ranking |
| `hallmark` | Local GMT overlap | Offline analysis, no internet required |

All modes output results as DataFrames and optionally save bar plots and dot plots. The MSigDB Hallmark 2020 library is used by default, targeting 16 cancer-relevant Hallmark gene sets including `HALLMARK_E2F_TARGETS`, `HALLMARK_P53_PATHWAY`, `HALLMARK_MYC_TARGETS_V1`, and `HALLMARK_APOPTOSIS`.

---

## Setup

### 1. Environment

```bash
conda create -n cancer-gnn python=3.8
conda activate cancer-gnn

# PyTorch (adjust cuda version)
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric

# All other dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Data is from the EMOGI benchmark (Schulte-Sasse et al., 2021). The six multi-omics HDF5 files must be downloaded from Zenodo and placed under `results/`.

**Step 1 — Download from Zenodo**

```bash
# EMOGI dataset (Zenodo record 3707301)
# Six HDF5 files totalling ~2 GB; requires an internet connection.
pip install zenodo-get   # or: pip install requests tqdm

# Option A: zenodo-get (recommended)
zenodo_get 3707301 -o ./zenodo_data

# Option B: direct wget (adjust URLs if the record changes)
mkdir -p ./zenodo_data
wget -P ./zenodo_data \
  "https://zenodo.org/record/3707301/files/CPDB_multiomics.h5" \
  "https://zenodo.org/record/3707301/files/IREF_multiomics.h5" \
  "https://zenodo.org/record/3707301/files/IREF_2015_multiomics.h5" \
  "https://zenodo.org/record/3707301/files/MULTINET_multiomics.h5" \
  "https://zenodo.org/record/3707301/files/PCNET_multiomics.h5" \
  "https://zenodo.org/record/3707301/files/STRINGdb_multiomics.h5"
```

**Step 2 — Place files in the expected layout**

```bash
mkdir -p results/EMOGI_CPDB results/EMOGI_IRefIndex results/EMOGI_IRefIndex_2015 \
         results/EMOGI_Multinet results/EMOGI_PCNet results/EMOGI_STRINGdb

cp ./zenodo_data/CPDB_multiomics.h5       results/EMOGI_CPDB/
cp ./zenodo_data/IREF_multiomics.h5       results/EMOGI_IRefIndex/
cp ./zenodo_data/IREF_2015_multiomics.h5  results/EMOGI_IRefIndex_2015/
cp ./zenodo_data/MULTINET_multiomics.h5   results/EMOGI_Multinet/
cp ./zenodo_data/PCNET_multiomics.h5      results/EMOGI_PCNet/
cp ./zenodo_data/STRINGdb_multiomics.h5   results/EMOGI_STRINGdb/
```

Final layout:
```
results/
├── EMOGI_CPDB/CPDB_multiomics.h5
├── EMOGI_IRefIndex/IREF_multiomics.h5
├── EMOGI_IRefIndex_2015/IREF_2015_multiomics.h5
├── EMOGI_Multinet/MULTINET_multiomics.h5
├── EMOGI_PCNet/PCNET_multiomics.h5
└── EMOGI_STRINGdb/STRINGdb_multiomics.h5
```

Each HDF5 file contains:
- `network` — adjacency matrix (dense)
- `features` — multi-omics node features (N × 64): MF, METH, GE, CNA × 16 cancer types
- `gene_names` — gene identifiers (Ensembl ID + HGNC symbol)
- `y_train`, `y_val`, `y_test` — cancer/non-cancer labels
- `mask_train`, `mask_val`, `mask_test` — train/val/test masks

**Verify download**

```bash
python -c "
import h5py, os
files = ['results/EMOGI_CPDB/CPDB_multiomics.h5',
         'results/EMOGI_IRefIndex/IREF_multiomics.h5',
         'results/EMOGI_IRefIndex_2015/IREF_2015_multiomics.h5',
         'results/EMOGI_Multinet/MULTINET_multiomics.h5',
         'results/EMOGI_PCNet/PCNET_multiomics.h5',
         'results/EMOGI_STRINGdb/STRINGdb_multiomics.h5']
for f in files:
    with h5py.File(f) as h:
        n = h['features'].shape[0]
        print(f'{os.path.basename(f)}: {n} genes, {h[\"features\"].shape[1]} features')
"
```

---

## Usage

### Run benchmark (Methodology 1)

```bash
# Test on CPDB (default)
python experiments/run_benchmark.py --gcn 1

# Test on each network (reproduce Table 1)
python experiments/run_benchmark.py --gcn 1 \
    --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB

python experiments/run_benchmark.py --gcn 1 \
    --dataset IREF_2015 IREF STRING PCNET CPDB MULTINET

# GAT / GIN variants
python experiments/run_benchmark.py --gat 1
python experiments/run_benchmark.py --gin 1

# MLP baseline (no graph structure)
python experiments/run_benchmark.py --mlp 1
```

### Run improved model (Methodologies 2 & 3)

```bash
# Default improved settings (GCN + all improvements)
python experiments/run_improved.py --gcn 1

# With explicit options
python experiments/run_improved.py --gcn 1 \
    --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \
    --use_residual True --use_batchnorm True \
    --use_net_weights True --lr_scheduler cosine \
    --label_smoothing 0.05 --normalize standard

# GraphSAGE backbone
python experiments/run_improved.py --sage 1

# Ablation: no residual connections
python experiments/run_improved.py --gcn 1 --use_residual False

# Ablation: no feature normalisation
python experiments/run_improved.py --gcn 1 --normalize none
```

### Hyperparameter search (Methodology 2)

```bash
python experiments/run_hparam_search.py \
    --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \
    --n_trials 50 --epochs_per_trial 500

# Persistent study (resumable)
python experiments/run_hparam_search.py \
    --n_trials 100 --study_name emgnn_search \
    --storage sqlite:///hparam_search.db
```

### Attribution analysis (Methodology 4)

```bash
# Feature importance for cancer genes
python experiments/run_attribution.py \
    --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \
    --gene_label cancer --node_explain

# Edge attribution for top predicted genes
python experiments/run_attribution.py \
    --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \
    --gene_label top_predicted --edge_explain --node_explain
```

### GSEA (Methodology 4)

```bash
# Enrichr ORA (requires internet)
python experiments/run_gsea.py \
    --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \
    --mode enrichr --top_n 200

# Pre-ranked GSEA using continuous prediction scores
python experiments/run_gsea.py \
    --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \
    --mode preranked

# Local Hallmark overlap (no internet)
python experiments/run_gsea.py \
    --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \
    --mode hallmark \
    --gmt_path ./data/h.all.v2023.1.Hs.symbols.gmt
```

---

## Key Parameters

### EMGNNImproved (`run_improved.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gcn/--gat/--gin/--sage` | `--gcn` | GNN backbone |
| `--hidden` | 64 | Hidden layer dimension |
| `--n_layers` | 3 | Number of GNN layers |
| `--dropout` | 0.5 | Dropout rate |
| `--lr` | 0.005 | Learning rate |
| `--epochs` | 2000 | Max training epochs |
| `--patience` | 250 | Early-stopping patience |
| `--use_residual` | True | Residual skip connections |
| `--use_batchnorm` | True | Deprecated: use `--norm_type` instead |
| `--norm_type` | batch | Normalisation: `batch`\|`graph`\|`layer`\|`none` — use `none` or `graph` for best results |
| `--use_net_weights` | True | Per-network importance weights |
| `--label_smoothing` | 0.05 | Label smoothing epsilon |
| `--lr_scheduler` | cosine | LR schedule: cosine/step/none |
| `--normalize` | standard | Feature norm: standard/minmax/none |
| `--feature_select` | True | Variance-threshold feature selection |
| `--seed` | 72 | Random seed for reproducibility |
| `--dataset` | all 6 | PPI networks (last = test set) |

---

## Output Artefacts

Training produces the following in `./results/my_models/<run_name>/`:

```
model.pkl                        Model weights (best validation epoch)
predictions.tsv                  Cancer probability for every gene
hyper_params.txt                 All hyperparameters for reproducibility
attribution/
  feature_importance_cancer.csv  Aggregated feature importance for cancer genes
  feature_importance_cancer.pdf  Importance bar plot (coloured by omics type)
  edge_attr_cancer_*.pkl         Per-gene edge attribution scores
gsea/
  enrichr_results.csv            Enrichr ORA significant terms
  enrichr_barplot.pdf            Enrichment bar plot
  preranked_results.csv          Pre-ranked GSEA results
  hallmark_overlap.csv           Hallmark overlap statistics
```

Global results are appended to:
- `./results/results.txt` (benchmark)
- `./results/results_improved.txt` (improved model)
- `./results/hparam_search_results.csv` (hyperparameter search)

---

## Results

Full experiment results are in [`results/experiment_summary.md`](results/experiment_summary.md).

### Quick summary

| Methodology | Key result |
|-------------|-----------|
| 1 — Reproduce benchmark | GCN: AUPR=0.748 (CPDB), 0.736 (STRING) — consistent with paper |
| 2 — Model optimisation | Residual connections maintain performance; BatchNorm hurts full-batch graphs |
| 3 — Multi-network (+3 PPI) | AUPR jumps from 0.748 → **0.788** (+4%) on CPDB |
| 4 — Interpretability | 31 significant cancer hallmark pathways (FDR<0.05); TP53/EGFR/BRCA1 top-ranked |

### Multi-network lift

| Networks | Model | AUPR | AUROC |
|---------|-------|------|-------|
| CPDB only | Benchmark GCN | 0.7479 | 0.8668 |
| IREF_2015 + MULTINET + CPDB | Benchmark GCN | **0.7877** | **0.9041** |
| IREF_2015 + MULTINET + CPDB | EMGNNImproved GCN | **0.7894** | 0.8989 |

### GSEA top pathways (top-200 predicted genes, MSigDB Hallmark 2020)

EMT (FDR=1.6×10⁻¹⁷), PI3K/AKT/mTOR (5.8×10⁻¹⁵), Apoptosis (6.4×10⁻¹¹), TGF-β (8.5×10⁻¹⁰), WNT (2.9×10⁻⁰⁸), p53 Pathway, G2-M Checkpoint, Hedgehog — 31 terms total.

---

## Citation

If you use this code, please cite the benchmark paper:

```bibtex
@article{chatzianastasis2023emgnn,
  author  = {Chatzianastasis, Michail and Vazirgiannis, Michalis and Zhang, Zijun},
  title   = {{Explainable Multilayer Graph Neural Network for Cancer Gene Prediction}},
  journal = {Bioinformatics},
  year    = {2023},
  volume  = {39},
  number  = {11},
  pages   = {btad643},
  doi     = {10.1093/bioinformatics/btad643}
}
```
