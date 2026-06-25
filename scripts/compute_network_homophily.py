#!/usr/bin/env python3
"""Compute label homophily for EMOGI/EMGNN HDF5 PPI networks on CPU.

The script uses the repository data loader registry by default. It requires the
HDF5 files under ``results/EMOGI_*``. It is safe to run on CPU and does not train
any model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))

from gcnIO import load_hdf_data  # noqa: E402
from src.data.loader import DATASET_PATHS  # noqa: E402


def labels_from_masks(y_train, y_val, y_test) -> np.ndarray:
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    if y_val is None:
        y_val = np.zeros_like(y_train)
    else:
        y_val = np.asarray(y_val).reshape(-1)
    return (y_train + y_val + y_test).astype(int)


def edge_pairs(adj) -> tuple[np.ndarray, np.ndarray]:
    if sp.issparse(adj):
        coo = adj.tocoo()
        rows, cols = coo.row, coo.col
    else:
        rows, cols = np.nonzero(np.asarray(adj))
    keep = rows < cols
    return rows[keep], cols[keep]


def homophily_metrics(labels: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> dict:
    edge_labels_u = labels[rows]
    edge_labels_v = labels[cols]
    labelled = (edge_labels_u >= 0) & (edge_labels_v >= 0)
    rows = rows[labelled]
    cols = cols[labelled]
    edge_labels_u = labels[rows]
    edge_labels_v = labels[cols]

    same = edge_labels_u == edge_labels_v
    positives = labels == 1
    negatives = labels == 0

    pos_nodes = np.flatnonzero(positives)
    neg_nodes = np.flatnonzero(negatives)
    incident_pos = np.isin(rows, pos_nodes) | np.isin(cols, pos_nodes)
    incident_neg = np.isin(rows, neg_nodes) | np.isin(cols, neg_nodes)
    pos_pos = (edge_labels_u == 1) & (edge_labels_v == 1)
    neg_neg = (edge_labels_u == 0) & (edge_labels_v == 0)

    return {
        "n_labelled_nodes": int(len(labels)),
        "n_positive": int(positives.sum()),
        "n_negative": int(negatives.sum()),
        "n_undirected_edges": int(len(rows)),
        "overall_edge_homophily": float(same.mean()) if len(same) else np.nan,
        "positive_edge_homophily": float(pos_pos.sum() / incident_pos.sum())
        if incident_pos.sum()
        else np.nan,
        "negative_edge_homophily": float(neg_neg.sum() / incident_neg.sum())
        if incident_neg.sum()
        else np.nan,
        "positive_incident_edge_count": int(incident_pos.sum()),
        "negative_incident_edge_count": int(incident_neg.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_PATHS),
        choices=list(DATASET_PATHS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/network_homophily_summary.csv"),
    )
    args = parser.parse_args()

    rows = []
    for name in args.datasets:
        path = Path(DATASET_PATHS[name])
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} HDF5 file at {path}. Download/copy the EMOGI "
                "benchmark data before running this CPU analysis."
            )
        adj, _features, y_train, y_val, y_test, *_rest = load_hdf_data(
            str(path), feature_name="features"
        )
        labels = labels_from_masks(y_train, y_val, y_test)
        src, dst = edge_pairs(adj)
        metrics = homophily_metrics(labels, src, dst)
        metrics["network"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    columns = ["network"] + [c for c in df.columns if c != "network"]
    df = df[columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

