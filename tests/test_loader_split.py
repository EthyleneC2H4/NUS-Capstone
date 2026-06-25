from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from src.data import loader as loader_module
from src.data.feature_engineering import FeatureEngineer


class LoaderSplitTests(unittest.TestCase):
    def setUp(self):
        self.original_loader = loader_module.load_hdf_data

        names = np.array([[str(i), f"G{i}"] for i in range(12)], dtype=object)
        features = np.arange(12 * 64, dtype=float).reshape(12, 64)
        adj = sparse.eye(12, format="csr")
        y_train = np.zeros((12, 1))
        y_train[:8] = np.arange(8).reshape(-1, 1) % 2
        y_val = np.zeros((12, 1))
        y_test = np.zeros((12, 1))
        y_test[8:] = np.arange(4).reshape(-1, 1) % 2
        train_mask = np.array([True] * 8 + [False] * 4)
        val_mask = np.zeros(12, dtype=bool)
        test_mask = ~train_mask
        feature_names = np.array(loader_module.FEATURES_ORDER, dtype=object)

        def fake_load(*_args, **_kwargs):
            return (
                adj,
                features,
                y_train,
                y_val,
                y_test,
                train_mask,
                val_mask,
                test_mask,
                names,
                feature_names,
            )

        loader_module.load_hdf_data = fake_load

    def tearDown(self):
        loader_module.load_hdf_data = self.original_loader

    def test_split_is_deterministic_and_round_trips(self):
        _loader, first = loader_module.load_multi_network_data(
            ["CPDB"], split_seed=72, val_fraction=0.25
        )
        _loader, second = loader_module.load_multi_network_data(
            ["CPDB"], split_seed=72, val_fraction=0.25
        )
        self.assertEqual(first["split_manifest"], second["split_manifest"])

        with tempfile.TemporaryDirectory() as tmp:
            split_path = Path(tmp) / "split.json"
            split_path.write_text(json.dumps(first["split_manifest"]))
            _loader, restored = loader_module.load_multi_network_data(
                ["CPDB"],
                split_seed=999,
                split_file=str(split_path),
                val_fraction=0.25,
            )

        for key in ("train", "validation", "test"):
            self.assertEqual(
                first["split_manifest"][key], restored["split_manifest"][key]
            )

    def test_feature_engineering_is_explicit_and_fitted(self):
        fe = FeatureEngineer(normalize="standard", feature_selection=False)
        _loader, info = loader_module.load_multi_network_data(
            ["CPDB"], feature_engineer=fe
        )
        self.assertTrue(fe.fitted)
        self.assertEqual(tuple(info["meta_x"].shape), (12, 64))


if __name__ == "__main__":
    unittest.main()

