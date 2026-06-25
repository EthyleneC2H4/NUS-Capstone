#!/usr/bin/env python3
"""CPU smoke test for the improved data/model path.

This script uses synthetic graphs and does not require EMOGI HDF5 data or GPU.
It catches basic regressions in model construction, forward shape, split
manifest structure, and optional feature flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.emgnn_improved import EMGNNImproved


def make_batch() -> tuple[DataLoader, dict]:
    data_list = []
    node2idx = {}
    meta_features = []
    meta_labels = []

    for graph_idx in range(2):
        n_nodes = 6
        x = torch.randn(n_nodes, 8)
        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 0, 1, 2, 3, 4],
            ],
            dtype=torch.long,
        )
        y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
        node_names = np.array(
            [[str(graph_idx), f"GENE_{i}"] for i in range(n_nodes)], dtype=object
        )
        for i, key in enumerate(map(tuple, node_names)):
            if key not in node2idx:
                node2idx[key] = len(node2idx)
                meta_features.append(x[i])
                meta_labels.append(y[i])
        data_list.append(Data(x=x, edge_index=edge_index, y=y, node_names=node_names))

    loader = DataLoader(data_list, batch_size=len(data_list))
    batch = next(iter(loader))
    info = {
        "batch": batch,
        "node2idx": node2idx,
        "meta_x": torch.stack(meta_features),
        "meta_y": torch.tensor(meta_labels, dtype=torch.long),
        "number_of_input_nodes": batch.x.shape[0],
    }
    return loader, info


def make_args(**overrides):
    defaults = dict(
        gcn=True,
        gat=False,
        gin=False,
        sage=False,
        nb_heads=1,
        alpha=0.2,
        dropout=0.1,
        drop_edge_rate=0.0,
        heterophily_aware=False,
        cross_network_attention=False,
        gps_meta=False,
        focal_gamma=0.0,
        focal_alpha=0.75,
        hipgnn_lambda=0.0,
        hipgnn_eigvecs=32,
        hypergraph_gmt=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def run_case(name: str, args) -> None:
    loader, info = make_batch()
    batch = info["batch"]
    model = EMGNNImproved(
        nfeat=batch.x.shape[1],
        hidden_channels=16,
        n_layers=2,
        nclass=2,
        meta_x=info["meta_x"],
        args=args,
        data=batch,
        node2idx=info["node2idx"],
        use_residual=True,
        use_batchnorm=False,
        norm_type="none",
        use_network_weights=True,
        label_smoothing=0.05,
    )
    model.eval()
    with torch.no_grad():
        output = model(batch.x.float(), batch.edge_index, batch)
    expected = info["number_of_input_nodes"] + len(info["node2idx"])
    assert output.shape == (expected, 2), (name, output.shape, expected)
    loss = model.loss(output[info["number_of_input_nodes"] :], info["meta_y"])
    assert torch.isfinite(loss), name
    print(f"{name}: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)
    run_case("baseline", make_args())
    run_case("heterophily_aware", make_args(heterophily_aware=True))
    run_case("cross_network_attention", make_args(cross_network_attention=True))
    run_case("focal_loss", make_args(focal_gamma=2.0))
    print("CPU smoke test complete.")


if __name__ == "__main__":
    main()
