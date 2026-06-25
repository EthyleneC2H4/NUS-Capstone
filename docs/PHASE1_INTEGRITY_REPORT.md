# Phase 1 Integrity Report

> Completed: 2026-06-14  
> Scope: Freeze and correct the current project state. No new scientific result
> is claimed by this phase.

## Completed Work

- Created local annotated tag `pre-integrity-audit-20260614` at commit `98fb298`.
- Established `NUS-Capstone/` as the authoritative deliverable repository.
- Added the current 13-page paper source and PDF to the repository working tree.
- Added independent `--split_seed` and optional `--split_file` arguments.
- Replaced unordered set slicing with deterministic validation sampling.
- Added full Python, NumPy, PyTorch, CUDA, and deterministic-algorithm seeding.
- Added per-run `split_manifest.json` containing gene-level train, validation,
  and test membership.
- Added per-run `experiment_metadata.json` containing command, arguments, Git
  state, environment, parameter count, split counts, and test metrics.
- Disabled previously inert feature-engineering flags by default and made them
  explicit opt-in operations fitted on first-network training nodes.
- Added `PROJECT_STATE.md` and `results/RESULT_PROVENANCE.md`.
- Corrected paper descriptions of preprocessing, network weighting, meta-GNN,
  classifier depth, interpretability model, and transductive evaluation.
- Removed the manual citation macro and restored keyed LaTeX citations.
- Corrected known reference metadata for EMGNN, SGCD, DISHyper, HIPGNN, DGHNN,
  and deepCDG.
- Corrected code/checkpoint availability claims and added author contribution,
  funding, conflict-of-interest, ethics, and AI-assistance declarations.
- Updated and copied the English and Chinese agent guides and project plans.

## Verification

- Python compilation passed for the modified loader and experiment entry point.
- Synthetic deterministic-split test passed.
- Saved split-manifest round-trip test passed.
- Explicit feature-engineering fit/transform test passed.
- `run_improved.py --help` exposes the new split controls.
- Paper compiled successfully from `NUS-Capstone/LaTeX/` in two passes.
- No unresolved citation warnings or visible `[?]` citations were found.
- Paper source and PDF copies are synchronised between the working directory
  and the authoritative repository.
- `git diff --check` passed.

## Important Boundary

Existing M1-M5 metrics remain legacy evidence because they predate the saved
split manifest. Phase 1 improves all future runs but does not retrospectively
make old model directories fully reproducible. As of the 2026-06-25 scope
decision, baseline/P7/P4 fixed-split reruns are optional future robustness
checks rather than required final deliverables. The final paper and README should
therefore label the current results as legacy evidence and disclose seed counts.

## CPU-Only Follow-Up on 2026-06-25

Additional non-GPU tasks were completed using existing files:

- Added reproducibility documentation and an environment file.
- Added unit tests and a synthetic CPU smoke test.
- Added a CPU M5 statistical-summary script and generated
  `results/m5_statistical_summary.csv`.
- Added a CPU network-homophily script for future use when EMOGI HDF5 data are
  available locally.

No GPU revalidation was run, and no absent-data analysis was marked complete.
Those experiments are outside the final required scope unless the project is
expanded later.

## Repository State

The Phase 1 changes are present but intentionally not committed. Review the
working tree and create one project commit after approval.
