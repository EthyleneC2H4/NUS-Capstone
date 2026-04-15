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
| 2 | Optimise model (residual, norm options, LR scheduler, label smoothing, hparam search) | ✅ `src/models/emgnn_improved.py` |
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

**Our reproduced results (RTX 5090, 2000 epochs, patience 250, seed 72)**

Single-network GCN — multiple runs:

| Network | Runs | Best AUPR | Best AUROC | Mean AUPR |
|---------|------|-----------|------------|-----------|
| CPDB    | 18   | 0.7528 | 0.8712 | 0.7432 |
| STRING  | 6    | 0.7588 | 0.8895 | 0.7391 |
| IREF_2015 | 2  | 0.7582 | 0.8795 | 0.7574 |
| MULTINET | 2   | 0.7835 | 0.9336 | 0.7760 |
| PCNET   | 2    | 0.7458 | 0.9296 | 0.7413 |
| IREF    | 2    | 0.6935 | 0.8968 | 0.6891 |

Multi-backbone comparison (CPDB):

| Backbone | Best AUPR | Best AUROC | Runs |
|----------|-----------|------------|------|
| GCN      | 0.7528    | 0.8712     | 18   |
| GIN      | **0.7918**| **0.8890** | 3    |
| GAT      | 0.6158    | 0.7790     | 3    |

GCN is stable and consistent across many runs (mean 0.743); GIN achieves higher peak AUPR but with fewer runs. GAT underperforms, consistent with the paper's observation that attention mechanisms need more data.

Multi-network benchmark:

| Networks | AUPR | AUROC | Δ AUPR |
|---------|------|-------|--------|
| CPDB only | 0.7479 | 0.8668 | — |
| IREF_2015 + MULTINET + CPDB | **0.7877** | **0.9041** | **+0.040** |

---

### Methodology 2 — Model Optimisation

**EMGNNImproved** (`src/models/emgnn_improved.py`) adds four improvements:

#### 2a. Residual connections (`use_residual=True`)
Skip connections are added from layer *l−1* to layer *l* for all GNN layers after the first:
```
h^(l) = LeakyReLU(GNN_l(h^(l-1))) + h^(l-1)
```
This improves gradient flow and enables training deeper networks without degradation.

#### 2b. Normalisation options (`norm_type`)

| `norm_type` | Implementation | Notes |
|-------------|---------------|-------|
| `batch`     | BatchNorm1d   | **Default but harmful** on full-batch graphs — see finding below |
| `graph`     | GraphNorm     | GNN-aware; normalises per-graph, recommended alternative |
| `layer`     | LayerNorm     | Normalises over feature dim; works for any batch size |
| `none`      | —             | No normalisation; best empirical performance here |

**Key finding:** BatchNorm1d hurts performance in full-batch graph learning. Because the entire graph is processed as one batch, running statistics are computed over all nodes simultaneously, providing no useful normalisation during inference. Use `--norm_type none` or `--norm_type graph` for best results.

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
norm_type ∈ {none, graph, layer}
label_smoothing ∈ [0, 0.2]
lr_scheduler ∈ {none, cosine, step}
```

**Ablation study results (GCN backbone, CPDB test set)**

| Configuration | AUPR | AUROC | Δ AUPR vs baseline |
|--------------|------|-------|-------------------|
| Benchmark (no improvements) | 0.7479 | 0.8668 | — |
| + Residual, no BN | 0.7440 | 0.8658 | −0.004 |
| + BN + Residual | 0.7061 | 0.8579 | −0.042 |
| + BN + Residual + CosineAnneal | 0.7064 | 0.8641 | −0.042 |
| + Residual + LabelSmooth(0.05), no BN | 0.7409 | 0.8600 | −0.007 |

**Optuna best hyperparameters (50 trials, CPDB):**

| Parameter | Best value |
|-----------|-----------|
| lr | 0.001758 |
| hidden | 32 |
| n_layers | 4 |
| dropout | 0.211 |
| weight_decay | 1.0×10⁻⁵ |
| use_residual | True |
| norm_type | none |
| label_smoothing | 0.003 |
| lr_scheduler | step |
| **Best AUPR** | **0.8023** |

---

### Methodology 3 — Multi-Network Extension

#### Learnable per-network importance weights (`use_network_weights=True`)
The original EMGNN treats all PPI networks equally. EMGNNImproved adds a learnable softmax-normalised scalar weight *w_k* per network:
```
w = softmax(θ)         # θ ∈ ℝ^K, K = number of networks
h_i ← h_i * w_{k(i)}  # weight node i's embedding by its network's importance
```
This allows the model to up-weight high-quality networks and down-weight noisy ones automatically during training.

**Multi-network results (GCN backbone)**

| Model | Networks | AUPR | AUROC | Δ vs M1 baseline |
|-------|---------|------|-------|----------------|
| Benchmark GCN | CPDB only | 0.7479 | 0.8668 | — |
| Benchmark GCN | IREF_2015 + MULTINET + CPDB | 0.7877 | 0.9041 | +0.040 |
| EMGNNImproved GCN | IREF_2015 + CPDB | 0.8018 | 0.9000 | +0.054 |
| EMGNNImproved GCN | IREF_2015 + MULTINET + CPDB | 0.7942 | 0.9018 | +0.046 |
| **EMGNNImproved GCN** | **All 6 networks** | **0.8067** | **0.9170** | **+0.059** |

**Learned per-network importance weights (6-network model, mean of 2 runs):**

| Network | Weight |
|---------|--------|
| CPDB | 0.2097 |
| MULTINET | 0.2005 |
| IREF_2015 | 0.1660 |
| STRING | 0.1659 |
| PCNET | 0.1617 |
| IREF | 0.0962 |

CPDB and MULTINET are consistently most important; IREF contributes least.

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
- **Node-feature attributions** — which of the 64 multi-omics features are most important per gene
- **Aggregated importance** — mean/max importance across a set of genes (e.g. all cancer genes), producing a global feature importance ranking
- **Edge attributions** — which meta-graph edges drive a given prediction (requires PyG ≤2.4)

**Top feature importance (Integrated Gradients, cancer genes):**

| Rank | Feature | Importance | Omics type |
|------|---------|------------|-----------|
| 1 | METH: LIHC | 0.908 | DNA methylation, liver cancer |
| 2 | GE: BLCA | 0.805 | Gene expression, bladder cancer |
| 3 | GE: BRCA | 0.778 | Gene expression, breast cancer |
| 4 | METH: CESC | 0.698 | DNA methylation, cervical cancer |
| 5 | METH: PRAD | 0.656 | DNA methylation, prostate cancer |
| 6 | GE: LIHC | 0.653 | Gene expression, liver cancer |
| 7 | METH: LUAD | 0.641 | DNA methylation, lung adenocarcinoma |
| 8 | MF: KIRP | 0.561 | Mutation frequency, kidney papillary |
| 9 | MF: BLCA | 0.549 | Mutation frequency, bladder cancer |
| 10 | GE: LUSC | 0.527 | Gene expression, lung squamous cell |

Methylation (METH) and gene expression (GE) features dominate, highlighting the role of epigenetic alterations and transcriptomic dysregulation as primary cancer signals.

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
conda create -n cancer-gnn python=3.10
conda activate cancer-gnn

# PyTorch + PyTorch Geometric (adjust for your CUDA version)
pip install torch torchvision torchaudio
pip install torch_geometric

# All other dependencies
pip install -r requirements.txt
```

For RTX 5090 (CUDA 12.8, sm_120):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### 2. Data Preparation

Data is from the EMOGI benchmark (Schulte-Sasse et al., 2021). The six multi-omics HDF5 files must be downloaded from Zenodo and placed under `results/`.

**Download from Zenodo (record 3707301)**

```bash
pip install zenodo-get
zenodo_get 3707301 -o ./zenodo_data
```

**Place files in the expected layout**

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

---

## Usage

### Run benchmark (Methodology 1)

```bash
# Test on CPDB (default)
python experiments/run_benchmark.py --gcn 1

# Test on each network individually
for net in CPDB STRING IREF_2015 IREF MULTINET PCNET; do
    python experiments/run_benchmark.py --gcn 1 --dataset $net
done

# GAT / GIN variants
python experiments/run_benchmark.py --gat 1
python experiments/run_benchmark.py --gin 1

# Multi-network
python experiments/run_benchmark.py --gcn 1 \
    --dataset IREF_2015 MULTINET CPDB
```

### Run improved model (Methodologies 2 & 3)

```bash
# Best configuration: all 6 networks, BN disabled
python experiments/run_improved.py --gcn 1 \
    --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \
    --norm_type none --use_residual True \
    --use_net_weights True --lr_scheduler cosine \
    --label_smoothing 0.05

# GraphSAGE backbone
python experiments/run_improved.py --sage 1 --norm_type none

# Ablation: no residual connections
python experiments/run_improved.py --gcn 1 --use_residual False --norm_type none
```

### Hyperparameter search (Methodology 2)

```bash
python experiments/run_hparam_search.py \
    --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \
    --n_trials 50 --epochs_per_trial 500

# Persistent study (resumable)
python experiments/run_hparam_search.py \
    --n_trials 100 --study_name emgnn_search \
    --storage sqlite:///results/hparam_search.db
```

### Attribution analysis (Methodology 4)

```bash
# Feature importance for cancer genes
python experiments/run_attribution.py \
    --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \
    --gene_label cancer --node_explain

# Feature importance for top predicted genes
python experiments/run_attribution.py \
    --model_dir ./results/my_models/EMGNNImproved_GCN_CPDB_... \
    --gene_label top_predicted --node_explain
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
| `--norm_type` | batch | Normalisation: `batch`\|`graph`\|`layer`\|`none` — use `none` for best results |
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
args.pkl                         Serialised training args (used by run_attribution.py)
attribution/
  feature_importance_cancer.csv  Aggregated feature importance for cancer genes
  feature_importance_cancer.pdf  Importance bar plot (coloured by omics type)
  feature_importance_top_predicted.csv / .pdf
gsea/
  enrichr_results.csv            Enrichr ORA significant terms
  enrichr_barplot.pdf            Enrichment bar plot
  preranked_results.csv          Pre-ranked GSEA results
  hallmark_overlap.csv           Hallmark overlap statistics
```

Global results are appended to:
- `./results/results.txt` — benchmark runs
- `./results/results_improved.txt` — improved model runs
- `./results/network_weights.txt` — learned per-network importance weights
- `./results/hparam_search_results.csv` — Optuna best hyperparameters

---

## Results

Full experiment results are in [`results/experiment_summary.md`](results/experiment_summary.md).

### Quick summary

| Methodology | Best Configuration | AUPR | Δ vs M1 | Key Finding |
|-------------|-------------------|------|---------|-------------|
| M1 — Reproduce | GCN/GIN, CPDB (multi-run) | 0.748–0.792 | — | Reproduces paper trend; GIN peak AUPR=0.792 |
| M1 — Multi-net | Benchmark GCN, 3 networks | 0.7877 | +0.040 | +4% from multi-network aggregation |
| M2 — Optimise | EMGNNImproved, Optuna | **0.8023** | +0.054 | BN harmful; optimal: hidden=32, n_layers=4, no BN |
| M3 — Multi-net | EMGNNImproved, 6 networks | **0.8067** | **+0.059** | +5.9%; CPDB+MULTINET most important |
| M4 — Interpret | Integrated Gradients + GSEA | — | — | 31 cancer pathways; methylation features dominant |

### Top predicted cancer genes

| Rank | Gene | P(cancer) | Known role |
|------|------|-----------|-----------|
| 1 | TP53 | 0.9999 | Master tumour suppressor |
| 2 | MUC16 | 0.9998 | CA-125 ovarian cancer biomarker |
| 3 | TTN | 0.9998 | Most frequently mutated gene in cancer |
| 4 | CTNNB1 | 0.9992 | β-catenin, WNT pathway |
| 5 | EP300 | 0.9990 | Histone acetyltransferase, tumour suppressor |
| 6 | PIK3CA | 0.9989 | PI3K catalytic subunit, breast cancer driver |
| 7 | FN1 | 0.9988 | Fibronectin, EMT marker |
| 8 | CREBBP | 0.9984 | Histone acetyltransferase |
| 9 | UBC | 0.9982 | Ubiquitin C, protein degradation hub |
| 10 | FAT4 | 0.9980 | Cadherin tumour suppressor |

### GSEA top pathways (top-200 predicted genes, MSigDB Hallmark 2020)

31 significant pathways (FDR < 0.05). Top enriched: EMT (FDR=1.6×10⁻¹⁷), PI3K/AKT/mTOR (5.8×10⁻¹⁵), Apoptosis (6.4×10⁻¹¹), TGF-β (8.5×10⁻¹⁰), WNT (2.9×10⁻⁰⁸), G2-M Checkpoint, Hedgehog, Notch — confirming biological plausibility of GNN predictions.

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
