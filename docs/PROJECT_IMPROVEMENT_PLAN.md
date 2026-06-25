# NUS Capstone Project Improvement Plan

> Project: Cancer Driver Gene Prediction via Heterophily-Aware Graph Neural Networks  
> Created: 2026-06-14  
> Purpose: Preserve the recommended roadmap for improving research validity, reproducibility, paper quality, and final delivery.

## 1. Guiding Principle

The immediate priority is not to add more model variants. The project should first align the implementation, experimental evidence, paper claims, and GitHub deliverables. Work should proceed in this order:

> **2026-06-25 update:** CPU-only follow-up work added tests, reproducibility
> documentation, an environment file, statistical summaries from existing M5
> rows, and a network-homophily script. GPU-dependent revalidation and analyses
> requiring absent HDF5 files are now optional future work.
>
> **2026-06-25 scope decision:** GPU fixed-split revalidation, direct homophily
> computation from absent HDF5 files, Focal/Label Smoothing interaction runs,
> and final-model IG/GSEA reruns are removed from the required graduation scope.
> They remain optional future work. The final project should use the existing
> results as explicitly labelled legacy evidence.

1. Correct research-integrity and implementation inconsistencies.
2. Scope the central experimental claims to existing evidence.
3. Keep final-model interpretability reruns as optional future work.
4. Make the repository reproducible.
5. Finalise and audit the paper.

## 2. Phase 1: Freeze and Correct the Current State

**Estimated duration:** 2-3 days  
**Priority:** Critical

### Tasks

- [x] Create a Git tag or branch such as `pre-integrity-audit` to preserve the current state.
- [x] Treat `NUS-Capstone/` as the authoritative project repository.
- [x] Add the current LaTeX source, experiment summaries, and reproducibility metadata to version control.
- [x] Replace the unordered `set`-based validation split with a deterministic, saved split.
- [x] Seed Python, NumPy, PyTorch, and CUDA consistently.
- [x] Record the exact train, validation, and test gene lists used by every core experiment.
- [x] Resolve the feature-engineering mismatch:
  - Current experiments did not actually pass `FeatureEngineer` into the loader.
  - First revise the paper to describe the implementation that produced the reported results.
  - Evaluate feature standardisation and feature selection later as explicit ablations.
- [x] Align the paper's architecture description with the implementation:
  - Number of meta-GNN layers.
  - Classifier depth.
  - Position at which network weights are applied.
  - Actual normalisation and preprocessing steps.
- [x] Replace the manual citation macro with BibTeX or another structured bibliography workflow.
- [x] Fix unresolved `[?]` citations, multi-source citations, DOI/page identifiers, and code/data availability claims.
- [x] Create a result-provenance table mapping every reported number to its CSV row, log, configuration, seed, Git commit, and model checkpoint.

### Exit Criteria

- Code, paper, and documentation describe the same pipeline.
- Data splits are deterministic and saved.
- Every major result has traceable evidence.
- The paper contains no unresolved citations.

## 3. Phase 2: Optional Robustness Revalidation

**Estimated duration:** 3-5 days, excluding server queue time  
**Priority:** Optional future work under the current scope

### Core Multi-Seed Experiments

If the project is expanded later, run at least five seeds using the same fixed
split:

- [ ] Six-network baseline.
- [ ] Six-network heterophily-aware model (P7).
- [ ] Six-network Cross-Network Attention model (P4).

For each configuration, report:

- Mean and standard deviation.
- 95% confidence interval.
- Per-seed AUPR and AUROC.
- Paired effect size and statistical test against the baseline.
- Training time and parameter count.

### Heterophily Evidence

- [ ] Calculate overall label homophily for every PPI network.
- [ ] Calculate class-specific homophily for cancer and non-cancer genes.
- [ ] Calculate label assortativity where appropriate.
- [ ] Analyse whether P7 improvements correlate with network heterophily.
- [ ] Add a table or figure showing the structural evidence supporting the central hypothesis.

### Optional Ablations

- [ ] Run a 2x2 experiment for Focal Loss and Label Smoothing:
  - Neither enabled.
  - Label Smoothing only.
  - Focal Loss only.
  - Both enabled.
- [ ] Repeat gain decomposition with multiple seeds:
  - Single-network benchmark.
  - Six-network benchmark.
  - Six-network improved baseline.
  - Six-network heterophily-aware model.
- [ ] Compare CPDB included versus excluded as an input when CPDB is the test network.
- [ ] Clearly distinguish the standard transductive protocol from a stricter leave-target-network-out protocol.

### Final-Scope Claim Policy

The original intended central claim was:

> Heterophily-aware gating consistently improves cancer driver gene prediction over an otherwise identical six-network baseline under fixed data splits and multiple random seeds.

For the final scoped submission, use the narrower claim that legacy three-seed
M5 results show P7 as the strongest evaluated extension over the six-network
baseline, with fixed-split confirmation left as future work. Do not claim current
best-in-class performance unless recent competing methods are reproduced on
identical datasets and splits.

### Exit Criteria

- The primary P7 result is supported by at least five paired runs.
- Heterophily is measured directly rather than inferred only from prior work.
- The Focal Loss explanation is experimentally tested.
- Data and architecture gains are separated under a consistent protocol.

## 4. Phase 3: Optional Interpretability Extensions

**Estimated duration:** 1-2 days  
**Priority:** Optional future work under the current scope

### Tasks

- [ ] Run Integrated Gradients on the final validated P7 model.
- [ ] Run Enrichr ORA and pre-ranked GSEA using predictions from the same final model.
- [ ] Compare at least two IG baselines:
  - Zero-vector baseline.
  - Feature-mean baseline.
- [ ] Create a 4 x 16 multi-omics feature-importance heatmap.
- [ ] Create a PPI-network weight or gene-level attention visualisation.
- [ ] Prepare case studies for:
  - TP53.
  - PIK3CA or CTNNB1.
  - One plausible newly predicted cancer gene.
- [ ] Explicitly discuss hub-gene and gene-length confounding, including TTN and UBC where relevant.
- [ ] Attempt GNNExplainer only after the final model and data pipeline are stable.

### Exit Criteria

- Predictive and interpretability results come from the same final model family.
- Attribution conclusions are not dependent on a single baseline choice.
- Biological interpretation includes both supporting evidence and confounding risks.

## 5. Phase 4: Reproducibility Engineering

**Estimated duration:** 2 days  
**Priority:** High

### Tasks

- [x] Save complete experiment configurations as YAML or JSON.
- [x] Save train, validation, and test splits as versioned files.
- [x] Add a locked environment specification (`environment.yml`, lock file, or pinned requirements).
- [x] Add lightweight tests for:
  - Dataset loading and split determinism.
  - Model forward shapes.
  - Network weighting.
  - Heterophily-aware gating.
  - Cross-network attention.
  - Optional feature flags.
- [x] Add a CPU smoke test using a small synthetic multi-network graph.
- [x] Provide one command that reproduces the main paper tables from saved result files.
- [ ] Provide one documented command for each core training experiment.
- [ ] Publish large checkpoints through GitHub Releases, Zenodo, or another suitable archive.
- [x] Remove or clearly mark obsolete result summaries and duplicated documentation.
- [x] Ensure the repository README points to files that actually exist.

### Exit Criteria

- A new user can install the environment and run the smoke test.
- Core experiments have complete commands and configurations.
- Public availability statements match the files that are actually published.
- Local and GitHub result summaries are identical.

## 6. Phase 5: Paper Revision and Final Audit

**Estimated duration:** 2-4 days  
**Priority:** Critical for submission

### Paper Revision

- [ ] Use the root `LaTeX/` directory as the paper working source unless a new authoritative location is explicitly chosen.
- [ ] Update the abstract using only the final scoped evidence.
- [ ] Rewrite Methods to match the exact code path and experimental protocol.
- [ ] Add seed counts, confidence where available, effect sizes where supported,
  and avoid direct dataset-level homophily claims unless measured.
- [ ] Separate validated explanations from hypotheses in Results and Discussion.
- [ ] Update Limitations to reflect the final experiment state.
- [ ] Correct the M4 model-description inconsistency.
- [ ] Add or verify:
  - Data Availability.
  - Code Availability.
  - Author Contributions.
  - Funding.
  - Conflict of Interest.
  - Ethics statement where applicable.
  - AI usage disclosure.
- [ ] Compile the final PDF and inspect every table, figure, citation, and hyperlink.
- [ ] Synchronise the final source and PDF into `NUS-Capstone/` and GitHub.

### Final Integrity Audit

- [ ] Verify every citation and DOI.
- [ ] Verify every numerical claim against source data.
- [ ] Verify that failed and pending experiments are disclosed.
- [ ] Verify that comparison protocols are described fairly.
- [ ] Run a final simulated academic review focused on methodology, evidence, and reproducibility.

### Exit Criteria

- No paper claim contradicts the code or experiment records.
- Every result is traceable to raw evidence.
- Local paper, repository paper, and README report the same status.
- The final PDF has no unresolved references or formatting defects.

## 7. Deferred Work

The following should not block the graduation deliverables unless the core phases finish early:

- P1 Random-Walk Positional Encoding optimisation.
- P3 lightweight or linear-attention GPS replacement.
- P5 pathway hypergraph experiments requiring MSigDB data.
- P6 HIPGNN experiments requiring spectral preprocessing.
- P8 PINNACLE embedding experiments.
- P10 GNNExplainer debugging.
- Large combinations of multiple M5 techniques.

These extensions increase scope but do not resolve the current validity and reproducibility risks.

## 8. Recommended Execution Order

1. Freeze the current Git state.
2. Fix deterministic splits and experiment metadata.
3. Reconcile paper Methods with the actual implementation.
4. Add tests, environment locking, and reproducibility commands.
5. Revise the paper so every claim matches the existing scoped evidence.
6. Mark fixed-split reruns, direct homophily measurement, Focal/Label Smoothing
   interaction ablation, CPDB leakage controls, and final-model IG/GSEA as
   optional future work rather than required tasks.
7. Perform the final integrity audit.
8. Synchronise all final changes to `NUS-Capstone/` and GitHub.

## 9. Definition of Done

The project is ready for final submission when all of the following are true:

- [ ] Every reported number maps to a raw log, configuration, split, seed, and commit.
- [ ] The main conclusion is worded as legacy existing evidence, with seed counts
  and provenance disclosed.
- [ ] Heterophily is presented as motivation supported by prior work unless a
  direct project-dataset measurement is later added.
- [ ] Paper Methods accurately describe the implemented model and preprocessing.
- [ ] The repository supports a clean smoke test and documented core reproduction path.
- [ ] No unresolved citation, unsupported best-in-class claim, or code-paper inconsistency remains.
- [ ] The local paper, GitHub PDF, README, and result summaries are synchronised.
