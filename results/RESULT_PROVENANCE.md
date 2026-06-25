# Result Provenance Index

> Audit date: 2026-06-14  
> Status: Final scoped evidence index. Existing numerical results are legacy
> unless explicitly labelled as fixed-split results.

## Evidence Rules

1. CSV and raw logs take precedence over narrative summaries.
2. A result is fully reproducible only when its model directory contains
   `split_manifest.json`, `experiment_metadata.json`, arguments, and predictions.
3. Legacy results may remain in the paper when clearly disclosed. Claims should
   report seed counts and avoid implying fixed-split validation unless those
   reruns are actually present.
4. Failed and skipped experiments are evidence about feasibility, not successful
   model evaluations.

## Core Result Map

| Claim or table | Reported value | Primary evidence | Status |
|---|---:|---|---|
| M1 CPDB GCN mean | AUPR 0.7432, 18 runs | `results/results.txt`; `results/experiment_summary.md` | Legacy |
| M1 CPDB GCN best | AUPR 0.7528 | `results/results.txt` | Legacy |
| M2 BatchNorm ablation | Delta AUPR -0.042 | `results/results_improved.txt`; `results/experiment_summary.md` | Legacy |
| M2 Optuna best | AUPR 0.8023, seed 72 | `results/hparam_search_results.csv` | Legacy, seed-dependent |
| M2 Optuna multi-seed | AUPR 0.7424 +/- 0.008 | `results/experiment_summary.md`; archived run records | Legacy |
| M3 benchmark six-network | AUPR 0.7987 | `results/results.txt`; `results/experiment_summary.md` | Legacy |
| M3 improved six-network best | AUPR 0.8067, AUROC 0.9170 | `results/results_improved.txt`; model archive | Legacy, single run |
| M3 fixed baseline | AUPR 0.8019 +/- 0.0050 | `results/m5_ablation_summary.csv` | Legacy five-seed summary |
| M4 Integrated Gradients | METH:LIHC 0.908 | archived model attribution CSV | Legacy six-network model |
| M4 GSEA | 28 Hallmark sets at FDR < 0.05 | archived `gsea/enrichr_results.csv` | Legacy six-network model |
| M5 P1 heterophily-aware | AUPR 0.8240 +/- 0.0044 | `results/m5_ablation_summary.csv` | Legacy three-seed final-scope result; optional fixed-split robustness check |
| M5 P2 cross-network attention | AUPR 0.8142 +/- 0.0053 | `results/m5_ablation_summary.csv` | Legacy three-seed final-scope result; optional fixed-split robustness check |
| M5 P3 DropEdge | AUPR 0.8064 +/- 0.0061 | `results/m5_ablation_summary.csv` | Legacy three-seed result |
| M5 P4 GraphMAE | AUPR 0.8002 +/- 0.0075 | `results/m5_ablation_summary.csv` | Legacy three-seed result |
| M5 P5 Focal Loss + label smoothing | AUPR 0.7608 +/- 0.0045 | `results/m5_ablation_summary.csv` | Legacy three-seed result; interaction explanation remains a hypothesis |
| Discarded GPS meta-encoder | CUDA OOM, about 189 GiB allocation request | archived local GPS logs | Failed |
| Discarded RWPE | impractical preprocessing cost at dimension 16 | archived local RWPE logs | Skipped/incomplete |
| Optional GNNExplainer | tensor-dimension mismatch | archived `gnnexplainer.log` | Failed/unresolved |

## Legacy M5 Raw Rows

The authoritative compact rows are stored in `results/m5_ablation_summary.csv`.
The five baseline seeds are 72, 1, 2, 42, and 99. P1, P2, P3, P4, and P5 use
seeds 72, 1, and 2.

`results/m5_statistical_summary.csv` was generated on 2026-06-25 by
`scripts/analyze_m5_results.py` without running any training. It reports:

| Experiment | n | Mean AUPR | Std AUPR | Mean AUROC | Delta AUPR | Welch p vs baseline | Paired p on matching seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_all | 5 | 0.80192 | 0.004996 | 0.91594 | 0.000000 | NA | NA |
| P1_heterophily | 3 | 0.82403 | 0.004356 | 0.91943 | 0.022113 | 0.00134 | 0.02953 |
| P2_cross_net_attn | 3 | 0.81417 | 0.005256 | 0.91763 | 0.012247 | 0.02977 | 0.10249 |
| P3_dropedge | 3 | 0.80640 | 0.006067 | 0.91583 | 0.004480 | 0.34686 | 0.33209 |
| P4_graphmae | 3 | 0.80017 | 0.007524 | 0.91553 | -0.001753 | 0.74277 | 0.93337 |
| P5_focal_loss | 3 | 0.76077 | 0.004456 | 0.89047 | -0.041153 | 9.04e-05 | 0.01343 |

## Requirements for New Results

Every new core run must retain:

- the exact command and parsed arguments;
- Git commit and dirty-state flag;
- package and CUDA versions;
- model and split seeds;
- complete gene-level train/validation/test split;
- per-seed metrics and model parameter count;
- raw log and prediction file.

New fixed-split results, if ever run as future work, should be added below this
section rather than replacing legacy rows.

## Optional Fixed-Split Results

No fixed-split reruns have been generated. They are outside the final required
scope as of 2026-06-25.

## Optional CPU Analyses Pending Data

`scripts/compute_network_homophily.py` can compute network-level and
class-specific homophily from the EMOGI HDF5 files. The files were not present in
the local workspace on 2026-06-25, so no homophily CSV was generated. Direct
dataset-level homophily measurements are therefore not part of the completed
evidence package.
