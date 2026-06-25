# AGENT_GUIDE.md — NUS Capstone Project Context Document

> **Purpose:** This document provides a complete project context for any AI agent working on this codebase. It covers background, architecture, current state, experimental results, environment, pending work, and actionable next steps.  
> **Last Updated:** 2026-06-14  
> **Author:** Yixi Wang (c2h4wang@u.nus.edu), NUS Department of Computer Science, Year 4 undergraduate

> **2026-06-14 integrity note:** Sections below preserve useful historical detail,
> but statements that M5 has not run are superseded. M5 evaluated five techniques;
> P7 heterophily-aware gating reached legacy three-seed AUPR 0.8240 +/- 0.0044.
> The authoritative current-state and evidence files are
> `NUS-Capstone/PROJECT_STATE.md` and
> `NUS-Capstone/results/RESULT_PROVENANCE.md`. Existing results predate the new
> deterministic split manifest. As of the 2026-06-25 scope decision, fixed-split
> GPU revalidation, direct homophily computation from absent HDF5 files, and
> final-model IG/GSEA reruns are optional future work rather than required
> graduation deliverables. Final claims should be scoped to the existing legacy
> evidence.

---

## Table of Contents

1. [Project Background](#1-project-background)
2. [Repository Structure](#2-repository-structure)
3. [Architecture & Core Code](#3-architecture--core-code)
4. [Data Pipeline](#4-data-pipeline)
5. [Experimental Results (Completed)](#5-experimental-results-completed)
6. [Improvement Proposals (P0–P10)](#6-improvement-proposals-p0p10)
7. [Paper Status](#7-paper-status)
8. [Runtime Environment](#8-runtime-environment)
9. [Pending / Unfinished Work](#9-pending--unfinished-work)
10. [Known Issues & Critical Findings](#10-known-issues--critical-findings)
11. [How to Run Experiments](#11-how-to-run-experiments)
12. [Key References](#12-key-references)
13. [Decision Log](#13-decision-log)
14. [Paper-to-Data Mapping: What Is Missing and Where](#14-paper-to-data-mapping-what-is-missing-and-where)

---

## 1. Project Background

### 1.1 Problem Statement

**Cancer driver gene prediction** using Graph Neural Networks (GNNs) on multiple Protein-Protein Interaction (PPI) networks with pan-cancer multi-omics data.

Cancer driver genes are genes whose mutations causally contribute to cancer development. Identifying them computationally is challenging because:
- **Label scarcity:** Only ~700 known cancer genes out of ~20,000 protein-coding genes (~3.5%)
- **Class imbalance:** 1:30 positive:negative ratio
- **Multiple data sources:** 6 independent PPI network databases provide complementary topological information
- **Heterophily motivation:** Prior work reports that PPI-based cancer driver prediction can be affected by low homophily; direct project-dataset homophily measurement is optional future work.

### 1.2 Base Method — EMGNN

This project extends **EMGNN** (Explainable Multilayer Graph Neural Network, Chatzianastasis et al., Bioinformatics 2023). EMGNN uses a two-stage architecture:

1. **Per-Network GNN Encoding:** Each PPI network is independently processed by a GNN (GCN/GIN/GAT) to produce node embeddings
2. **Meta-Graph Construction:** Node embeddings from all networks are mapped to unique gene-level "meta-nodes" (union of nodes across networks)
3. **Meta-Graph GNN + Classifier:** A second GCN operates on the meta-graph, followed by an MLP classifier for binary cancer/non-cancer prediction

### 1.3 Five Methodologies

| ID | Name | Objective |
|----|------|-----------|
| **M1** | Benchmark Reproduction | Reproduce EMGNN results across GCN/GIN/GAT backbones and 6 PPI networks |
| **M2** | Model Optimisation | Ablation study + Bayesian hyperparameter search (Optuna, 50 trials) |
| **M3** | Multi-Network Extension | Learnable per-network importance weights across all 6 PPI databases |
| **M4** | Interpretability | Integrated Gradients feature attribution + GSEA pathway enrichment analysis |
| **M5** | Advanced Techniques | 11 modular extensions (P0–P10): Focal Loss, PE, GraphMAE, GPS, CrossNetAttn, HyperGNN, HIPGNN, PINNACLE, DropEdge, Heterophily, GNNExplainer |

### 1.4 Six PPI Networks

| Network | Description | Nodes (approx.) | Source |
|---------|-------------|------------------|--------|
| CPDB | ConsensusPathDB | ~15K | Consensus of curated databases |
| IREF_2015 | iRefIndex 2015 | ~13K | Literature-curated interactions |
| IREF | iRefIndex (latest) | ~17K | Literature-curated interactions |
| STRING | STRINGdb | ~18K | Combined score from multiple evidence channels |
| PCNET | Parsimonious Composite Network | ~19K | Integrated network with quality filter |
| MULTINET | MultiNet | ~14K | Multi-evidence network |

### 1.5 Node Features

64-dimensional multi-omics features per gene, organized as 4 omics types × 16 TCGA cancer types:

| Feature Group | Abbreviation | Description | Dimensions |
|---------------|-------------|-------------|------------|
| Mutation Frequency | MF | Somatic mutation rate per cancer type | 16 |
| DNA Methylation | METH | Promoter methylation (beta values) | 16 |
| Gene Expression | GE | RNA-seq expression levels | 16 |
| Copy Number Alteration | CNA | Copy number changes | 16 |

---

## 2. Repository Structure

**Root:** `/Users/ethylene/Learning/NUS/Sem2/Capstone/capstone/`
**GitHub:** `git@github.com:EthyleneC2H4/NUS-Capstone.git`

```
capstone/                              # Project root
├── NUS-Capstone/                      # Main codebase (git-tracked, pushed to GitHub)
│   ├── src/                           # Source modules
│   │   ├── models/
│   │   │   ├── emgnn_improved.py      # ★ Core model (21.5 KB, ~452 lines)
│   │   │   ├── hypergnn.py            # HyperGNN for pathway encoding (P5)
│   │   │   ├── hipgnn.py              # HIPGNN anomaly detection (P6)
│   │   │   └── baselines.py           # Simple GCN/MLP baselines
│   │   ├── data/
│   │   │   ├── loader.py              # Multi-network data loader (224 lines)
│   │   │   ├── feature_engineering.py # Z-score, PCA, feature selection
│   │   │   ├── build_hypergraph.py    # GMT → hypergraph incidence matrix (P5)
│   │   │   └── pinnacle_embeddings.py # Load PINNACLE protein embeddings (P8)
│   │   ├── training/
│   │   │   ├── trainer.py             # Training loop with LR scheduler + early stopping
│   │   │   ├── hparam_search.py       # Optuna Bayesian hyperparameter search
│   │   │   └── pretrain_graphmae.py   # GraphMAE self-supervised pretraining (P2)
│   │   └── explainability/
│   │       ├── attribution.py         # Integrated Gradients (Captum)
│   │       └── gsea.py                # GSEA: Enrichr ORA + preranked + Hallmark overlap
│   │
│   ├── experiments/                   # Runnable scripts
│   │   ├── run_benchmark.py           # M1: Original EMGNN benchmark
│   │   ├── run_improved.py            # ★ M2/M3/M5: Main training script (40+ CLI args)
│   │   ├── run_hparam_search.py       # M2: Optuna search driver
│   │   ├── run_attribution.py         # M4: Feature attribution analysis
│   │   ├── run_gsea.py                # M4: Gene set enrichment analysis
│   │   ├── run_gnn_explain.py         # P10: GNNExplainer edge-level explanations
│   │   ├── run_m5_ablation.sh         # ★ M5 ablation: 8 experiments × 3 seeds
│   │   └── run_m5_explain.sh          # P10: GNNExplainer wrapper script
│   │
│   ├── benchmark/                     # Original EMGNN code (reference, read-only)
│   │   ├── model.py                   # Original EMGNN/GCN/MLP models
│   │   ├── train.py                   # Original training script
│   │   ├── explain.py                 # Original Integrated Gradients
│   │   ├── gcnIO.py                   # HDF5 I/O utilities
│   │   └── captum_custom.py           # Captum edge/node mask wrappers
│   │
│   ├── configs/
│   │   ├── benchmark_config.yaml      # M1 hyperparameters
│   │   └── improved_config.yaml       # M2/M3 optimised hyperparameters
│   │
│   ├── results/                       # Experiment outputs (mostly git-ignored)
│   │   ├── my_models/                 # ~60 trained model directories (git-ignored)
│   │   ├── experiment_summary.md      # ★ All results in one place
│   │   ├── hparam_search_results.csv  # Optuna best trials
│   │   ├── results.txt                # Benchmark results
│   │   ├── results_improved.txt       # Improved model results
│   │   └── network_weights.txt        # Learned per-network weights
│   │
│   ├── LaTeX/
│   │   ├── main.pdf                   # ★ Compiled paper (committed to git)
│   │   ├── main.tex                   # LaTeX source (git-ignored)
│   │   ├── sections/*.tex             # Paper sections (git-ignored)
│   │   └── figures/                   # Figures (enrichr_barplot.pdf)
│   │
│   ├── requirements.txt               # Python dependencies
│   ├── README.md                      # Project documentation
│   ├── .gitignore                     # Excludes .pkl, .h5, .tex, model dirs
│   └── LICENSE
│
├── EMGNN/                             # Original EMGNN reference repo (separate)
├── results/                           # Local copy of experiment outputs
├── LaTeX/                             # Local LaTeX working copy
├── logs/                              # Documentation & progress notes
│   ├── experiment_summary.md
│   ├── future_improvements.md         # ★ 11 improvement proposals (P0-P10) in detail
│   ├── progress_assessment_2026-05-07.md # ★ Methodology evaluation + gap analysis
│   ├── implementation_guide.md
│   ├── server_run_guide.md
│   └── ...
├── config/
│   └── config_autodl.md               # AutoDL server SSH credentials
├── mid-term/                          # Mid-term report & presentation
└── ref/                               # Reference papers (PDFs)
```

### Key .gitignore Rules

- `*.pkl`, `*.h5`, `*.npz` — binary data files excluded
- `results/my_models/` — trained model directories excluded (too large)
- `LaTeX/*.tex`, `LaTeX/sections/` — only PDF committed
- `*.sh` excluded EXCEPT `experiments/run_m5_*.sh`

---

## 3. Architecture & Core Code

### 3.1 EMGNNImproved (`src/models/emgnn_improved.py`)

The core model with 11 modular improvement flags:

```
Input: 6 PPI graphs × (nodes, 64-dim features, edge_index)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼       (× 6 networks)
   [Input Linear Projection]            nfeat → hidden_channels
        │
   [GCN/GIN/GAT/SAGE Layers × n_layers]
   • Optional: residual connections
   • Optional: norm_type (batch|graph|layer|none)
   • Optional: DropEdge (P9)
   • Optional: heterophily-aware gating (P7)
        │
   [Per-Network Weighting]
   • Scalar softmax weights (default)
   • OR CrossNetworkAttention (P4): gene-level attention via scatter_softmax
        │
        └──────→ Meta-Graph Construction
                 (union of genes across networks)
                    │
             [Meta-Graph GNN]
             • Standard GCNConv (default)
             • OR GPS Graph Transformer (P3)
                    │
             [Optional: Hypergraph Fusion (P5)]
             • HyperGNNEncoder on GO/KEGG gene sets
             • Fuse via learned gate: h_final = gate * h_meta + (1-gate) * h_hyper
                    │
             [MLP Classifier → log_softmax]
                    │
             [Loss: NLL / Focal Loss (P0)]
             • + label smoothing (optional)
             • + HIPGNN auxiliary loss (P6, multi-task)
```

### 3.2 Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `EMGNNImproved` | `emgnn_improved.py` | Main model with all improvement flags |
| `FocalLoss` | `emgnn_improved.py` | Class-imbalanced loss (gamma, alpha) |
| `CrossNetworkAttention` | `emgnn_improved.py` | Gene-level per-network attention |
| `HighLowPassSeparation` | `emgnn_improved.py` | Heterophily-aware gated fusion |
| `HyperGNNEncoder` | `hypergnn.py` | Multi-layer hypergraph encoder |
| `HIPGNNAuxHead` | `hipgnn.py` | Spectral + spatial anomaly detection |
| `GraphMAE` | `pretrain_graphmae.py` | Masked autoencoder for pretraining |
| `Trainer` | `trainer.py` | Training loop with LR scheduler, early stopping, gradient clipping |
| `FeatureEngineer` | `feature_engineering.py` | Z-score/MinMax scaling, variance selection, optional PCA |

### 3.3 CLI Arguments (`experiments/run_improved.py`)

**Backbone selection:** `--gcn`, `--gat`, `--gin`, `--sage`, `--mlp` (boolean flags)

**Dataset:** `--dataset IREF_2015 IREF STRING PCNET MULTINET CPDB` (last = test set)

**Architecture flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--use_residual` | True | Skip connections |
| `--norm_type` | batch | batch/graph/layer/none |
| `--use_net_weights` | True | Learnable per-network importance |
| `--label_smoothing` | 0.05 | Soft targets epsilon |
| `--cross_network_attention` | False | P4: gene-level attention |
| `--heterophily_aware` | False | P7: gated high/low-pass |
| `--focal_gamma` | 0.0 | P0: Focal Loss gamma (0 = off) |
| `--focal_alpha` | 0.75 | P0: positive class weight |
| `--pe_dim` | 0 | P1: Random-Walk PE dimensions |
| `--drop_edge_rate` | 0.0 | P9: edge dropout fraction |
| `--gps_meta` | False | P3: GPS Transformer for meta-graph |
| `--gps_heads` | 4 | P3: number of attention heads |
| `--pretrain_graphmae` | False | P2: enable GraphMAE pretraining |
| `--pretrain_epochs` | 200 | P2: pretraining epoch count |
| `--hypergraph_gmt` | None | P5: path to GMT file for hypergraph |
| `--hipgnn_lambda` | 0.0 | P6: auxiliary anomaly loss weight |
| `--hipgnn_eigvecs` | 32 | P6: Laplacian eigenvector count |
| `--pinnacle_path` | None | P8: path to .npz protein embeddings |

**Training hyperparameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--lr` | 0.005 | Learning rate |
| `--weight_decay` | 5e-4 | L2 regularization |
| `--hidden` | 64 | Hidden channels |
| `--n_layers` | 3 | GNN depth |
| `--dropout` | 0.5 | Dropout rate |
| `--epochs` | 2000 | Max training epochs |
| `--patience` | 250 | Early stopping patience |
| `--lr_scheduler` | cosine | cosine/step/none |
| `--seed` | 72 | Random seed |

---

## 4. Data Pipeline

### 4.1 Data Source

Multi-omics node features stored in HDF5 format from the EMOGI benchmark (Zenodo).

**Path on server:** `/root/NUS-Capstone/results/EMOGI_*/`

| File | Network |
|------|---------|
| `EMOGI_CPDB/CPDB_multiomics.h5` | CPDB |
| `EMOGI_IRefIndex/IREF_multiomics.h5` | IREF |
| `EMOGI_IRefIndex_2015/IREF_2015_multiomics.h5` | IREF_2015 |
| `EMOGI_Multinet/MULTINET_multiomics.h5` | MULTINET |
| `EMOGI_PCNet/PCNET_multiomics.h5` | PCNET |
| `EMOGI_STRINGdb/STRINGdb_multiomics.h5` | STRING |

Each HDF5 contains:
- `network`: Adjacency matrix (dense or sparse)
- `features`: Node features (64-dim multi-omics)
- `node_names`: Gene symbol identifiers
- `y_train`, `y_test`: Binary labels

### 4.2 Loading Flow (`src/data/loader.py`)

```
load_multi_network_data(dataset_names, ...)
  │
  ├─ For each network:
  │   ├─ Read HDF5 → features, adjacency, labels
  │   ├─ Sparse cache: convert dense adjacency → scipy.sparse → .npz (first run only)
  │   ├─ Feature alignment: reorder to canonical [MF, METH, GE, CNA] × 16 types
  │   ├─ Optional: FeatureEngineer.fit_transform()
  │   ├─ Optional: AddRandomWalkPE (pe_dim)
  │   └─ Build PyG Data(x, edge_index, y) with self-loops
  │
  ├─ Build node2idx: {(db_id, gene_symbol): unique_meta_idx}
  ├─ Build meta_x: (n_unique_genes, feat_dim) — average features across networks
  ├─ Build meta_y: (n_unique_genes,) — binary labels
  ├─ Split: train (last-1 graphs) / val (10% of train) / test (last graph)
  │
  └─ Return: DataLoader, info dict
```

### 4.3 External Data (Not Yet Available)

| Data | Source | Used by | Status |
|------|--------|---------|--------|
| GMT gene sets | MSigDB (Hallmark/C2) | P5 HyperGNN | **Not downloaded** |
| Laplacian eigenvectors | Computed from graph | P6 HIPGNN | **Not precomputed** |
| PINNACLE embeddings | HuggingFace / Nature Methods 2024 | P8 PINNACLE | **Not downloaded** |

---

## 5. Experimental Results (Completed)

### 5.1 M1 — Benchmark Reproduction

| Network | Best AUPR | Best AUROC | Mean AUPR | Runs |
|---------|-----------|------------|-----------|------|
| CPDB | **0.7528** | 0.8712 | 0.7432 | 18 |
| STRING | 0.7588 | 0.8895 | 0.7391 | 6 |
| IREF_2015 | 0.7582 | 0.8795 | 0.7574 | 2 |
| MULTINET | 0.7835 | **0.9336** | 0.7760 | 2 |
| IREF | 0.6935 | 0.8968 | 0.6891 | 2 |
| PCNET | 0.7458 | 0.9296 | 0.7413 | 2 |

Multi-backbone on CPDB: GIN (0.7918) > GCN (0.7528) > GAT (0.6158)
Multi-network (3-net benchmark GCN): AUPR 0.7877, AUROC 0.9041

### 5.2 M2 — Model Optimisation

**Ablation findings:**
- BatchNorm1d is **harmful** in full-batch graph learning: -4.2% AUPR
- Residual connections + no BatchNorm recommended
- Label smoothing (epsilon=0.05) has marginal effect

**Optuna best (50 trials, single seed=72):**
- AUPR = **0.8023** (hidden=32, n_layers=4, dropout=0.211, no BN, step LR)
- **WARNING:** Multi-seed validation (seeds 1-5) shows mean 0.7424 ± 0.008 — the 0.8023 is seed-dependent and NOT robust

### 5.3 M3 — Multi-Network Extension

| Configuration | AUPR | AUROC | Notes |
|---------------|------|-------|-------|
| Benchmark GCN, CPDB only | 0.7479 | 0.8668 | Baseline |
| Benchmark GCN, all 6 networks | **0.7987** | **0.9114** | Data effect only: +0.054 |
| EMGNNImproved, CPDB only | 0.7540 | 0.8615 | Architecture adds +0.006 |
| EMGNNImproved, 2-net (IREF_2015+CPDB) | 0.8018 | 0.9000 | Near-best with just 2 nets |
| **EMGNNImproved, all 6 networks** | **0.8067** | **0.9170** | **Best overall: +5.9%** |

**Decomposition:** Data contributes +0.054 AUPR; architecture contributes only +0.008.

**Learned network weights (softmax):**
- CPDB: 0.210, MULTINET: 0.201, IREF_2015: 0.166, STRING: 0.166, PCNET: 0.162, IREF: 0.096

### 5.4 M4 — Interpretability

**Top features (Integrated Gradients):**
1. METH:LIHC (0.908), 2. GE:BLCA (0.805), 3. GE:BRCA (0.778)
- Methylation + Gene Expression dominate (8/10 top features)

**Top predicted genes:** TP53, MUC16, TTN, CTNNB1, EP300, PIK3CA — all known cancer drivers

**GSEA:** 28 significant Hallmark pathways (FDR < 0.05):
- EMT (FDR=1.6×10^-33), PI3K/AKT/mTOR, Apoptosis, WNT, TGF-beta

### 5.5 M5 — Advanced Techniques

**Status: Code implemented, experiments NOT YET RUN.**

All 11 improvements (P0-P10) are coded and committed. The ablation experiment script (`run_m5_ablation.sh`) is ready to run. Server code has been synced and verified.

---

## 6. Improvement Proposals (P0–P10)

### Implemented & Ready for Ablation (no external data needed)

| ID | Name | CLI Flag | Difficulty | Expected AUPR Gain |
|----|------|----------|------------|---------------------|
| P0 | Focal Loss | `--focal_gamma 2.0 --focal_alpha 0.75` | Low | +2-4% |
| P1 | Random-Walk PE | `--pe_dim 16` | Low | +1-3% |
| P2 | GraphMAE Pretraining | `--pretrain_graphmae 1 --pretrain_epochs 200` | Medium | +3-5% |
| P3 | GPS Graph Transformer | `--gps_meta 1 --gps_heads 4` | Medium | +2-5% |
| P4 | Cross-Network Attention | `--cross_network_attention 1` | Medium | +1-3% |
| P7 | Heterophily-Aware | `--heterophily_aware 1` | Medium | +2-4% |
| P9 | DropEdge | `--drop_edge_rate 0.1` | Low | +0.5-1.5% |
| P10 | GNNExplainer | separate script | Low | N/A (interpretability) |

### Require External Data (deferred)

| ID | Name | CLI Flag | Required Data | Status |
|----|------|----------|---------------|--------|
| P5 | Hypergraph (GO/Pathway) | `--hypergraph_gmt <path>` | MSigDB GMT files | Not downloaded |
| P6 | HIPGNN Anomaly Detection | `--hipgnn_lambda 0.1 --hipgnn_eigvecs 32` | Precomputed Laplacian eigenvectors | Not computed |
| P8 | PINNACLE Embeddings | `--pinnacle_path <path>` | PINNACLE .npz from HuggingFace | Not downloaded |

---

## 7. Paper Status

**Title:** "Cancer Driver Gene Prediction via Explainable Multilayer Graph Neural Networks with Multi-Network Integration and Pathway Enrichment Analysis"

**Location:** `LaTeX/main.pdf` (17 pages, single-column, A4)

**Current section structure (revised 2026-05-30):**

| # | File | Section | Status |
|---|------|---------|--------|
| 0 | `00_frontmatter.tex` | Title, Authors, Abstract (~200 words), Keywords | ✅ Complete |
| 1 | `01_introduction.tex` | Introduction with 5 numbered contributions | ✅ Complete |
| 2 | `02_related_work.tex` | Related Work + comparison Table 1 | ✅ Complete |
| 3 | `03_methods.tex` | Methods (merged theory+methodology, all math, M1–M5 methods, **Fig. 1 architecture diagram**) | ✅ Complete |
| 4 | `04_results.tex` | Results: M1–M4 data (7 tables, 1 GSEA barplot figure), M5 implementation status, Discussion | ⚠️ M5 ablation table is **placeholder** |
| 5 | `05_conclusions.tex` | Conclusions + 6 named limitations | ✅ Complete |
| 6 | `06_declarations.tex` | Acknowledgements, Funding, Ethics, Code/Data Availability | ✅ Complete |
| 7 | `07_references.tex` | 33 numbered references (manual cite system) | ✅ Complete |

**Figures:** 2 (Fig. 1: TikZ architecture diagram in methods; Fig. 2: GSEA enrichment barplot in results)

**Remaining paper gaps:**
- [ ] M5 ablation results table — placeholder in `04_results.tex` (see Section 14 below)
- [ ] Multi-seed variance for M3 6-network result — only single-seed reported
- [ ] Additional visualisations (feature importance heatmap, training curves) — nice-to-have
- [ ] Re-run IG/GSEA on 6-network model — current M4 results are from single-network model (known limitation noted in paper)

---

## 8. Runtime Environment

### 8.1 Local Machine

- macOS Darwin 24.6.0, Apple Silicon
- Python 3.10+, PyTorch 2.2.2 (CPU)
- Used for development, paper writing, code management

### 8.2 Cloud Server (AutoDL)

**Connection:**
```bash
ssh -p 32197 root@connect.westd.seetacloud.com
# Password: aNeGrlDbcwUu
```

**Specs:**
- GPU: NVIDIA RTX 5090 (CUDA 12.8, sm_120)
- OS: Ubuntu, PyTorch 2.7, PyG 2.7
- Conda environment: `cancer-gnn`
- Python path: `/root/miniconda3/envs/cancer-gnn/bin/python`
- Project path: `/root/NUS-Capstone/`

**Important notes:**
- Server charges per-minute when running — **shut down immediately after experiments**
- Shutdown command: `shutdown now` (from SSH)
- **GitHub mirror (ghproxy.com) is unreliable** on the server — use SCP for code sync instead of git pull
- Server code was last synced via SCP on 2026-05-22 (all 11 files verified)
- No-card mode available (for non-GPU prep work at lower cost)

### 8.3 SCP Code Sync Workflow

When GitHub is unavailable on server:
```bash
# From local machine:
sshpass -p 'aNeGrlDbcwUu' scp -P 32197 \
  /Users/ethylene/Learning/NUS/Sem2/Capstone/capstone/NUS-Capstone/src/models/emgnn_improved.py \
  root@connect.westd.seetacloud.com:/root/NUS-Capstone/src/models/emgnn_improved.py
```

### 8.4 Dependencies

```
torch==2.2.2 (local CPU) / 2.7 (server GPU)
torch-geometric==2.7.0
captum==0.8.0
numpy==1.26.4
scipy==1.12.0
scikit-learn==1.7.2
pandas==2.2.2
h5py==3.11.0
matplotlib==3.8.4
optuna==4.8.0
gseapy==1.1.13
```

---

## 9. Pending / Unfinished Work

### 9.1 High Priority — M5 Ablation Experiments

**Status:** Script ready, server code synced, NOT YET RUN.

The ablation script `experiments/run_m5_ablation.sh` tests 8 experiments × 3 seeds (72, 1, 2):

| Experiment | Improvement | Extra Flags |
|------------|-------------|-------------|
| baseline | None (6-network reference) | — |
| P0_focal_loss | Focal Loss | `--focal_gamma 2.0 --focal_alpha 0.75` |
| P9_dropedge | DropEdge | `--drop_edge_rate 0.1` |
| P7_heterophily | Heterophily-Aware | `--heterophily_aware 1` |
| P2_graphmae | GraphMAE Pretraining | `--pretrain_graphmae 1 --pretrain_epochs 200` |
| P1_pe_dim16 | Positional Encoding | `--pe_dim 16` |
| P3_gps_meta | GPS Transformer | `--gps_meta 1 --gps_heads 4` |
| P4_cross_net_attn | Cross-Network Attention | `--cross_network_attention 1` |

All experiments share base config: GCN backbone, all 6 networks, norm_type=none, residual=True, cosine LR, label_smoothing=0.05, hidden=64, n_layers=3.

**Estimated runtime:** ~8-9 hours on RTX 5090.

**Command to run:**
```bash
cd /root/NUS-Capstone
nohup bash experiments/run_m5_ablation.sh > results/m5_ablation.log 2>&1 &
```

**Output:** `results/m5_ablation_summary.csv` (experiment, seed, aupr, auroc, duration)

### 9.2 Medium Priority — GNNExplainer (P10)

After ablation, run on best model:
```bash
bash experiments/run_m5_explain.sh results/my_models/<best_model_dir>
```

### 9.3 Medium Priority — Paper Update

After M5 experiments complete:
1. Add M5 ablation table to paper (Section 4: Results)
2. Update abstract with M5 findings
3. Add architecture diagram (currently only 1 figure)
4. Update conclusions section
5. Recompile PDF and push to GitHub

### 9.4 Lower Priority — External Data Experiments

| Task | Data Needed | How to Obtain |
|------|-------------|---------------|
| P5 Hypergraph | MSigDB GMT files | Download from msigdb.org (Hallmark + C2 collections) |
| P6 HIPGNN eigenvectors | Precomputed from graph Laplacian | Call `precompute_laplacian_eigvecs()` from `hipgnn.py` |
| P8 PINNACLE | 128-dim protein embeddings (.npz) | Download from HuggingFace (Li et al. Nature Methods 2024) |

### 9.5 Deliverables Not Yet Prepared

- [ ] Final presentation slides (mid-term slides exist at `mid-term/`)
- [ ] Codebase zip for submission
- [ ] Re-run IG/GSEA on 6-network best model (current M4 results are from single-network model)

---

## 10. Known Issues & Critical Findings

### 10.1 BatchNorm is Harmful

**Finding:** BatchNorm1d causes -4.2% AUPR degradation in full-batch graph learning.

**Reason:** In full-batch training, the entire graph is one "batch." Running statistics computed over the whole graph provide no meaningful normalization during inference. The model trains with graph-wide statistics but infers with running-mean statistics that don't generalize.

**Action:** Always use `--norm_type none` or `--norm_type layer`. Never use `--norm_type batch` or `--use_batchnorm True`.

### 10.2 Optuna Result is Seed-Dependent

The Optuna-found best configuration (AUPR=0.8023 at seed=72) drops to mean 0.7424 ± 0.008 across seeds 1-5. This indicates overfitting to a specific train/val split, NOT a robust hyperparameter improvement.

### 10.3 Multi-Network Gain is Mostly Data, Not Architecture

Benchmark GCN on 6 networks achieves AUPR=0.7987. EMGNNImproved adds only +0.008. The +5.9% total gain is mostly from integrating more PPI data.

### 10.4 CPDB Circular Dependency

CPDB is both an input network AND the test set. The high learned weight for CPDB (0.210) may be inflated by this circular dependency.

### 10.5 Server GitHub Mirror Unreliable

`mirror.ghproxy.com` times out on the AutoDL server. Use SCP for code sync instead of `git pull`.

### 10.6 Conda Path in Non-Interactive Shell

Shell scripts on the server must use the full conda Python path (`/root/miniconda3/envs/cancer-gnn/bin/python`) because `nohup` runs in non-interactive mode where conda is not activated.

---

## 11. How to Run Experiments

### 11.1 M5 Ablation (Main Pending Experiment)

```bash
# 1. Start AutoDL server with GPU
# 2. SSH into server
ssh -p 32197 root@connect.westd.seetacloud.com

# 3. Run ablation
cd /root/NUS-Capstone
nohup bash experiments/run_m5_ablation.sh > results/m5_ablation.log 2>&1 &

# 4. Monitor progress
tail -f results/m5_ablation.log

# 5. After completion (~8-9 hours), results in:
cat results/m5_ablation_summary.csv

# 6. Shut down server to stop billing
shutdown now
```

### 11.2 Single Experiment Run

```bash
/root/miniconda3/envs/cancer-gnn/bin/python experiments/run_improved.py \
  --gcn 1 \
  --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \
  --norm_type none --use_residual True --use_net_weights True \
  --lr_scheduler cosine --label_smoothing 0.05 \
  --hidden 64 --n_layers 3 --dropout 0.5 \
  --epochs 2000 --patience 250 \
  --seed 72 \
  --focal_gamma 2.0 --focal_alpha 0.75  # Example: enable Focal Loss
```

### 11.3 GNNExplainer

```bash
bash experiments/run_m5_explain.sh results/my_models/<model_directory>
```

### 11.4 GSEA Analysis

```bash
python experiments/run_gsea.py \
  --model_dir results/my_models/<model_dir> \
  --mode enrichr \
  --top_n 200
```

---

## 12. Key References

| Paper | Year | Venue | Relevance |
|-------|------|-------|-----------|
| EMOGI (Chatzianastasis et al.) | 2023 | Bioinformatics | **Base method** — multilayer GNN |
| Focal Loss (Lin et al.) | 2017 | ICCV | P0 — class imbalance handling |
| GraphMAE (Hou et al.) | 2022 | KDD | P2 — self-supervised pretraining |
| GPS (Rampasek et al.) | 2022 | NeurIPS | P3 — Graph Transformer architecture |
| R-GCN (Schlichtkrull et al.) | 2018 | ESWC | P4 reference — relational graph convolution |
| DISHyper (Deng et al.) | 2024 | Bioinformatics (ISMB) | P5 reference — disease hypergraph |
| HIPGNN | 2025 | AAAI | P6 reference — anomaly detection perspective |
| SGCD | 2024 | Briefings in Bioinformatics | P7 reference — heterophily in PPI |
| PINNACLE (Li et al.) | 2024 | Nature Methods | P8 — pretrained protein embeddings |
| DropEdge (Rong et al.) | 2020 | ICLR | P9 — graph-structure regularization |
| GNNExplainer (Ying et al.) | 2019 | NeurIPS | P10 — edge-level explanations |
| DGHNN (Li et al.) | 2025 | Bioinformatics | Recent comparator, GNN + HyperGNN + FT-Transformer |
| deepCDG | 2025 | Briefings in Bioinformatics | Recent competitor, cross-omics attention |

---

## 13. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-04 | Use GCN as primary backbone | Most stable across seeds (18 runs characterized) |
| 2026-04-15 | Disable BatchNorm | Harmful in full-batch GNN: -4.2% AUPR |
| 2026-05-07 | Prioritize G1 (re-run IG/GSEA on 6-net model) | Highest impact paper improvement |
| 2026-05-21 | Implement P0-P10 as modular flags | Enables clean ablation study without code branches |
| 2026-05-22 | Use SCP instead of git pull on server | GitHub mirror unreliable on AutoDL |
| 2026-05-22 | Hardcode conda Python path in scripts | Non-interactive shell cannot activate conda |
| 2026-05-22 | Exclude LaTeX source from GitHub | Only PDF needed; source is local-only |
| 2026-05-28 | M5 ablation tests 8 experiments × 3 seeds | Balances coverage vs. compute cost |

---

## Appendix A: Model Checkpoints

Key saved model directories on server (`/root/NUS-Capstone/results/my_models/`):

| Directory | Description |
|-----------|-------------|
| `GCN_['IREF_2015', 'IREF', 'STRING', 'PCNET', 'MULTINET', 'CPDB']_2026_05_09_09_52_59` | Best 6-network model (AUPR=0.8067) |
| `EMGNNImproved_GCN_CPDB_2026_04_15_10_22_58` | M4 interpretability model (single-network) |

Each directory contains: `model.pkl`, `args.pkl`, `batch.pkl`, `node2idx.pkl`, `predictions.tsv`, `hyper_params.txt`, `final_y.pkl`, `edge_index.pkl`

## Appendix B: Git Commit History

```
c4d9945 Add M5 ablation experiment scripts for RTX 5090 server
0931429 Remove LaTeX source files from repo, keep only PDF
2050bd4 Update paper with Methodology 5: eleven advanced GNN techniques
e178447 Add LaTeX paper source and PDF to repository
d1650b3 P8: Add PINNACLE pretrained protein embedding support
91df9cd P6: Add HIPGNN-inspired anomaly detection auxiliary head
d65fb1f P5: Add GO/Pathway hypergraph integration
c721a9b P4: Add cross-network attention fusion for gene-level network weighting
3e84f1a P3: Add GPS Graph Transformer option for meta-graph layer
35dfcab P1: Add Random-Walk positional encoding for graph-position awareness
1f25253 P10: Add GNNExplainer for edge-level interpretability
7967d24 P2: Add GraphMAE self-supervised pretraining
c133ffb P7: Add heterophily-aware gated high/low-pass separation
7e27bdd P9: Add DropEdge regularization for graph-structure robustness
07fe2b4 P0: Add Focal Loss for class-imbalanced cancer gene prediction
7b965a9 Add G2/G4 validation results and update GSEA to 6-network model
...
3b45caf Initial commit
```

## 14. Paper-to-Data Mapping: What Is Missing and Where

> **Purpose:** This section explicitly maps every piece of missing experimental data to its
> exact location in the LaTeX paper, so that once experiments are run, the paper can be
> updated directly without re-reading the entire document.

### 14.1 M5 Ablation Results Table (HIGH PRIORITY)

| Item | Detail |
|------|--------|
| **What's missing** | Ablation results for 8 M5 techniques × 3 random seeds |
| **Paper file** | `LaTeX/sections/04_results.tex` |
| **Paper location** | Section 4.5 "Advanced GNN Techniques: Implementation Status" — search for `%% PLACEHOLDER: M5 ABLATION RESULTS TABLE` |
| **What to do** | 1. Uncomment the `\begin{table}` block below the PLACEHOLDER marker. 2. Fill in Mean AUPR, std, Mean AUROC, and ΔAUPR for each row. 3. Update the preceding paragraph to discuss actual results instead of "has not yet been executed." |
| **Data source** | `results/m5_ablation_summary.csv` (generated by `run_m5_ablation.sh`) |
| **Server command** | `cd /root/NUS-Capstone && nohup bash experiments/run_m5_ablation.sh > results/m5_ablation.log 2>&1 &` |
| **Estimated runtime** | ~8–9 hours on RTX 5090 |
| **Also update** | (a) Abstract (`00_frontmatter.tex`): mention M5 top result if significant gain found. (b) Conclusions (`05_conclusions.tex`): update "M5 evaluation pending" limitation. (c) Discussion (`04_results.tex`, Section 4.6): add M5 discussion sub-section. |

### 14.2 Multi-Seed Variance for M3 Best Result (MEDIUM PRIORITY)

| Item | Detail |
|------|--------|
| **What's missing** | Multi-seed (seeds 1–5) AUPR ± std for the 6-network EMGNNImproved model |
| **Paper file** | `LaTeX/sections/04_results.tex` |
| **Paper location** | Section 4.3, Table 4 (`tab:multinetwork`) — the row "EMGNNImproved, All 6 networks" currently shows single-seed result |
| **What to do** | Run 5 seeds, compute mean ± std, update the table and add a footnote. Also update conclusions limitation "Single-seed evaluation" in `05_conclusions.tex`. |
| **Server command** | Run `run_improved.py` with `--seed 1`, `--seed 2`, ..., `--seed 5` using the 6-network config |

### 14.3 GNNExplainer Edge-Level Results (MEDIUM PRIORITY)

| Item | Detail |
|------|--------|
| **What's missing** | GNNExplainer results on best M5 model (or best M3 model) |
| **Paper file** | `LaTeX/sections/04_results.tex` |
| **Paper location** | Currently no GNNExplainer results section exists. Add as Section 4.4.4 "Edge-Level Interpretability via GNNExplainer" after the GSEA subsection. |
| **What to do** | 1. Run GNNExplainer. 2. Create a new subsection showing top edge importances for selected cancer genes. 3. Reference `\cite{gnnexplainer2019}`. |
| **Server command** | `bash experiments/run_m5_explain.sh results/my_models/<best_model_dir>` |
| **Also update** | (a) Abstract: mention edge-level interpretability. (b) Methods Section 3.5.11: confirm the method description matches actual output. |

### 14.4 External Data Modules (LOWER PRIORITY)

These three M5 techniques require external datasets not yet available on the server:

#### P5: Pathway Hypergraph (GO/KEGG)

| Item | Detail |
|------|--------|
| **What's missing** | MSigDB GMT files (Hallmark + C2 collections) |
| **How to obtain** | Download from https://www.gsea-msigdb.org/gsea/downloads.jsp (requires free registration) |
| **Server path** | Place at `/root/NUS-Capstone/data/msigdb_hallmark.gmt` |
| **Run command** | Add `--hypergraph_gmt data/msigdb_hallmark.gmt` to training command |
| **Paper update** | Add P5 row to M5 ablation table if results obtained; update Table 7 (`tab:m5_summary`) status from ◇ to ✓ |

#### P6: HIPGNN Eigenvectors

| Item | Detail |
|------|--------|
| **What's missing** | Precomputed Laplacian eigenvectors for each PPI network |
| **How to obtain** | Call `precompute_laplacian_eigvecs()` from `src/models/hipgnn.py` |
| **Run command** | Add `--hipgnn_lambda 0.1 --hipgnn_eigvecs 32` to training command |
| **Paper update** | Same as P5 |

#### P8: PINNACLE Embeddings

| Item | Detail |
|------|--------|
| **What's missing** | 128-dim protein embeddings (.npz) from PINNACLE |
| **How to obtain** | Download from HuggingFace (Li et al., Nature Methods 2024) |
| **Server path** | Place at `/root/NUS-Capstone/data/pinnacle_embeddings.npz` |
| **Run command** | Add `--pinnacle_path data/pinnacle_embeddings.npz` to training command |
| **Paper update** | Same as P5 |

### 14.5 Additional Visualisations (NICE-TO-HAVE)

| Figure | Paper location | Data source | How to create |
|--------|---------------|-------------|---------------|
| Feature importance heatmap (4 omics × 16 cancer types) | Add after Table 5 (`tab:featimp`) in Section 4.4.1 | IG attribution scores from M4 model | Run `run_attribution.py`, use matplotlib heatmap |
| Network weights bar chart | Add after Table 4 (`tab:netweights`) in Section 4.3 | Learned weights from 6-network model | Simple bar chart of the 6 weights |
| Training loss/AUPR curves | Add to Section 4.3 or Appendix | Training logs from server | Parse training log, plot with matplotlib |

### 14.6 Quick Reference: Paper File → Missing Data Summary

| Paper File | What's Missing | Priority |
|-----------|---------------|----------|
| `00_frontmatter.tex` | Update abstract with M5 results (if significant) | After M5 |
| `04_results.tex` Sec 4.5 | M5 ablation table (PLACEHOLDER marked) | **HIGH** |
| `04_results.tex` Sec 4.3 | Multi-seed variance for 6-net result | MEDIUM |
| `04_results.tex` Sec 4.4 | GNNExplainer results subsection | MEDIUM |
| `04_results.tex` Sec 4.6 | M5 discussion sub-section | After M5 |
| `05_conclusions.tex` | Update "M5 pending" and "single-seed" limitations | After M5 |
| `07_references.tex` | No new references needed unless new methods cited | — |

---

## Appendix C: Quick Start for New Agents

1. **Read this document first** for full context
2. **Core model:** `NUS-Capstone/src/models/emgnn_improved.py`
3. **Main training script:** `NUS-Capstone/experiments/run_improved.py`
4. **All results:** `NUS-Capstone/results/experiment_summary.md`
5. **Improvement proposals:** `logs/future_improvements.md`
6. **Progress evaluation:** `logs/progress_assessment_2026-05-07.md`
7. **Server config:** `config/config_autodl.md`
8. **Immediate task:** Finalise paper and repository claims around existing legacy evidence; keep fixed-split reruns as optional future work
9. **Paper-to-data mapping:** See Section 14 above for exactly what data goes where
