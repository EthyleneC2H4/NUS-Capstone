#!/usr/bin/env python3
"""Summarise existing M5 ablation results without running training.

This script consumes ``results/m5_ablation_summary.csv`` and emits:

- per-experiment means and standard deviations;
- deltas against the baseline mean;
- Welch tests against all available baseline seeds;
- paired tests against baseline seeds when matching seeds are available.

It intentionally treats the CSV as legacy evidence. It does not claim Phase 2
fixed-split validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def summarise(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = df[(df["aupr"] != "FAILED") & (df["auroc"] != "FAILED")].copy()
    df["aupr"] = df["aupr"].astype(float)
    df["auroc"] = df["auroc"].astype(float)
    df["summary_experiment"] = df["experiment"].where(
        ~df["experiment"].str.startswith("baseline"), "baseline_all"
    )

    baseline = df[df["summary_experiment"] == "baseline_all"]
    baseline_mean = baseline["aupr"].mean()

    rows = []
    for experiment, group in df.groupby("summary_experiment", sort=False):
        aupr = group["aupr"].to_numpy(dtype=float)
        auroc = group["auroc"].to_numpy(dtype=float)
        delta = aupr.mean() - baseline_mean

        welch_p = np.nan
        if experiment != "baseline_all":
            welch_p = stats.ttest_ind(
                aupr,
                baseline["aupr"].to_numpy(dtype=float),
                equal_var=False,
            ).pvalue

        paired_p = np.nan
        paired_baseline = baseline[baseline["seed"].isin(group["seed"])]
        paired_group = group[group["seed"].isin(paired_baseline["seed"])]
        if (
            experiment != "baseline_all"
            and len(paired_baseline) >= 2
        ):
            base_by_seed = paired_baseline.set_index("seed").loc[
                paired_group["seed"], "aupr"
            ]
            paired_p = stats.ttest_rel(
                paired_group["aupr"].to_numpy(dtype=float),
                base_by_seed.to_numpy(dtype=float),
            ).pvalue

        rows.append(
            {
                "experiment": experiment,
                "n": int(len(group)),
                "mean_aupr": aupr.mean(),
                "std_aupr": aupr.std(ddof=1) if len(aupr) > 1 else np.nan,
                "mean_auroc": auroc.mean(),
                "std_auroc": auroc.std(ddof=1) if len(auroc) > 1 else np.nan,
                "delta_aupr_vs_baseline": delta,
                "welch_p_vs_baseline": welch_p,
                "paired_p_vs_matching_baseline_seeds": paired_p,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/m5_ablation_summary.csv"),
        help="M5 ablation CSV produced by the legacy experiment scripts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/m5_statistical_summary.csv"),
        help="Output CSV for statistical summary.",
    )
    args = parser.parse_args()

    summary = summarise(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
