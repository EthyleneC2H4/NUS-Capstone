# Authoritative Project State

> Effective date: 2026-06-14

## Source of Truth

`NUS-Capstone/` is the authoritative repository for code, experiment scripts,
paper source, paper PDF, and compact result summaries.

- `src/` and `experiments/`: maintained implementation and experiment entry points.
- `LaTeX/`: authoritative paper source and compiled PDF.
- `results/RESULT_PROVENANCE.md`: mapping from paper claims to experiment evidence.
- `results/m5_ablation_summary.csv`: compact M5 result table.
- `results/m5_statistical_summary.csv`: CPU-generated statistical summary from
  existing M5 rows.
- `COMPLETE_EXPERIMENT_RESULTS.md`: historical narrative summary. Entries marked as
  pending or legacy must not be treated as newly validated evidence.

The parent-directory copies under `../LaTeX/` and `../results/` are working or
archival copies. Final changes must be synchronised into this repository before
they are considered part of the deliverable.

## Experiment Generations

### Legacy experiments

Experiments run before the Phase 1 integrity update used the historical split
implementation. Their reported metrics remain useful project evidence, but their
model directories do not contain a JSON split manifest or complete environment
metadata. They must not be silently combined with new fixed-split experiments.

### Fixed-split experiments

Experiments run after the Phase 1 update use:

- a model seed supplied by `--seed`;
- an independent split seed supplied by `--split_seed` (default `72`);
- a deterministic validation sample from sorted eligible genes;
- optional exact split reuse through `--split_file`;
- global Python, NumPy, PyTorch, and CUDA seeding;
- `split_manifest.json` and `experiment_metadata.json` in each model directory.

Feature engineering is disabled by default to preserve the preprocessing that
produced the legacy results. It is applied only when explicitly requested and is
fitted on first-network training nodes.

## Current Research Status

- M1-M4 and the June 2026 M5 runs are the final scoped experimental evidence.
  They remain labelled as legacy evidence because they predate the split-manifest
  protocol, and the paper/README should disclose seed counts and provenance.
- Fixed-split reruns of baseline, P7, and P4 would strengthen robustness, but
  they are no longer required deliverables for the current graduation scope.
- The CPU homophily script is available as an optional utility, but the local
  workspace currently lacks the EMOGI HDF5 files required to compute network
  homophily directly. Direct homophily measurements should therefore not be
  presented as completed project evidence.
- P5, P6, P8, P10, large-scale P1/P3 variants, fixed-split GPU revalidation,
  Focal/label-smoothing interaction runs, and final-model IG/GSEA reruns are
  removed from the required final scope and retained only as optional future work.

## Synchronisation Rule

Any code, result, or paper change is complete only when the corresponding files
inside `NUS-Capstone/` are updated and `git status` shows the intended changes.

## CPU-Only Completion on 2026-06-25

The remaining non-GPU tasks were advanced using existing result files:

- Added CPU smoke tests and unit tests.
- Added `scripts/analyze_m5_results.py` and generated
  `results/m5_statistical_summary.csv`.
- Added `scripts/compute_network_homophily.py` for later CPU homophily analysis
  once HDF5 data are present.
- Added `environment.yml` and `docs/REPRODUCIBILITY.md`.

GPU-dependent revalidation and analyses requiring absent HDF5 files are not part
of the final required scope. They remain useful optional future extensions, but
the current deliverable should be judged from the existing, explicitly labelled
evidence.
