"""
Attribution analysis for EMGNN / EMGNNImproved.

Wraps Captum's IntegratedGradients to produce:
1. Edge attributions  – which meta-graph edges are most important for a prediction
2. Node-feature attributions – which of the 64 multi-omics features matter most

This is a clean rewrite of benchmark/explain.py that:
- Works with both the original EMGNN and EMGNNImproved
- Separates data loading from attribution computation
- Returns tidy DataFrames for downstream analysis / plotting

Dependencies
------------
    pip install captum
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

# Re-use captum wrappers from benchmark
import sys
sys.path.insert(0, str(Path(__file__).parents[2] / 'benchmark'))
from captum_custom import to_captum  # noqa: E402

FEATURES_ORDER: List[str] = (
    [f'MF: {c}' for c in
     ['UCEC','BLCA','THCA','KIRC','READ','LUAD','ESCA','LUSC',
      'BRCA','COAD','HNSC','KIRP','PRAD','LIHC','STAD','CESC']]
  + [f'METH: {c}' for c in
     ['UCEC','BLCA','THCA','KIRC','READ','LUAD','ESCA','LUSC',
      'BRCA','COAD','HNSC','KIRP','PRAD','LIHC','STAD','CESC']]
  + [f'GE: {c}' for c in
     ['UCEC','BLCA','THCA','KIRC','READ','LUAD','ESCA','LUSC',
      'BRCA','COAD','HNSC','KIRP','PRAD','LIHC','STAD','CESC']]
  + [f'CNA: {c}' for c in
     ['UCEC','BLCA','THCA','KIRC','READ','LUAD','ESCA','LUSC',
      'BRCA','COAD','HNSC','KIRP','PRAD','LIHC','STAD','CESC']]
)


class AttributionAnalyzer:
    """
    Compute and aggregate Integrated-Gradients attributions for EMGNN.

    Parameters
    ----------
    model : trained EMGNN / EMGNNImproved (on CUDA)
    batch : PyG Batch object (all graphs concatenated)
    meta_x : meta-node feature tensor
    meta_edge_index : meta-graph edge index
    number_of_input_nodes : int
    all_node_names : array of node name strings
    final_y : label tensor for all nodes
    device : str
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch,
        meta_x: torch.Tensor,
        meta_edge_index: torch.Tensor,
        number_of_input_nodes: int,
        all_node_names: np.ndarray,
        final_y: torch.Tensor,
        device: str = 'cuda',
    ):
        self.model = model
        self.batch = batch
        self.meta_x = meta_x
        self.meta_edge_index = meta_edge_index
        self.number_of_input_nodes = number_of_input_nodes
        self.all_node_names = all_node_names
        self.final_y = final_y
        self.device = device

        # Build name → index map (meta-nodes only)
        self.name_to_meta_idx: Dict[str, int] = {}
        for i, name in enumerate(all_node_names[number_of_input_nodes:]):
            self.name_to_meta_idx[name.replace('_Meta_Node', '')] = i

    # ──────────────────────────────────────────────────────────────────────────

    def _output_idx(self, meta_idx: int) -> int:
        """Convert meta-node local index to absolute node index."""
        return self.number_of_input_nodes + meta_idx

    # ──────────────────────────────────────────────────────────────────────────
    # Edge attribution
    # ──────────────────────────────────────────────────────────────────────────

    def edge_attribution(
        self, meta_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Integrated-Gradients edge attributions for a single meta-node.

        Returns
        -------
        (ig_scores, normalised_scores) – both shape (n_meta_edges,)
        """
        output_idx = self._output_idx(meta_idx)
        target = int(self.final_y[output_idx])

        captum_model = to_captum(self.model, mask_type='edge',
                                  output_idx=output_idx)
        edge_mask = torch.ones(
            self.meta_edge_index.shape[1],
            requires_grad=True, device=self.device
        )
        ig = IntegratedGradients(captum_model)
        ig_attr = ig.attribute(
            edge_mask.unsqueeze(0),
            target=target,
            additional_forward_args=(
                self.batch.x.to(self.device),
                self.batch.edge_index.to(self.device),
                self.batch,
                self.meta_edge_index.to(self.device),
                None, True,
            ),
            internal_batch_size=1,
        )
        raw = ig_attr.squeeze(0).abs().cpu().detach().numpy()
        normed = raw / (raw.max() + 1e-9)
        return raw, normed

    # ──────────────────────────────────────────────────────────────────────────
    # Node-feature attribution
    # ──────────────────────────────────────────────────────────────────────────

    def node_feature_attribution(
        self, meta_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Integrated-Gradients node-feature attributions.

        Returns
        -------
        (ig_scores, normalised_scores) – both shape (n_features,)
        The scores are summed across all nodes (meta-node + input nodes)
        to give a global feature importance signal.
        """
        output_idx = self._output_idx(meta_idx)
        target = int(self.final_y[output_idx])

        captum_model = to_captum(self.model, mask_type='node',
                                  output_idx=output_idx)
        ig = IntegratedGradients(captum_model)
        x_all = torch.cat(
            [self.batch.x.to(self.device),
             self.meta_x.to(self.device)],
            dim=0
        ).float()
        ig_attr, _ = ig.attribute(
            x_all.unsqueeze(0),
            target=target,
            additional_forward_args=(
                self.batch.edge_index.to(self.device),
                self.batch,
                None,
                self.batch.x.to(self.device),
                True,
            ),
            internal_batch_size=1,
            return_convergence_delta=True,
        )
        # Sum over nodes, keep feature dimension
        feat_scores = ig_attr.squeeze(0).abs().sum(dim=0).cpu().detach().numpy()
        normed = feat_scores / (feat_scores.max() + 1e-9)
        return feat_scores, normed

    # ──────────────────────────────────────────────────────────────────────────
    # Batch attribution + aggregation
    # ──────────────────────────────────────────────────────────────────────────

    def aggregate_feature_importance(
        self,
        meta_indices: List[int],
        mode: str = 'mean',
    ) -> pd.DataFrame:
        """
        Compute and aggregate feature importances over multiple genes.

        Parameters
        ----------
        meta_indices : list of meta-node local indices
        mode : 'mean' or 'max'

        Returns
        -------
        DataFrame with columns: feature | importance
        """
        all_scores = []
        for idx in meta_indices:
            _, normed = self.node_feature_attribution(idx)
            all_scores.append(normed)

        arr = np.stack(all_scores, axis=0)  # (n_genes, n_features)
        agg = arr.mean(axis=0) if mode == 'mean' else arr.max(axis=0)

        df = pd.DataFrame({
            'feature': FEATURES_ORDER[:len(agg)],
            'importance': agg,
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        return df

    def plot_feature_importance(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
        title: str = 'Feature Importance (Integrated Gradients)',
        save_path: Optional[str] = None,
    ):
        """Bar plot of top-N features by importance."""
        import matplotlib.pyplot as plt

        top = df.head(top_n)
        # Colour by omics type
        colour_map = {'MF': '#4C72B0', 'METH': '#DD8452',
                      'GE': '#55A868', 'CNA': '#C44E52'}
        colours = [colour_map.get(f.split(':')[0].strip(), 'grey')
                   for f in top['feature']]

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35 + 1)))
        ax.barh(range(len(top)), top['importance'], color=colours)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['feature'], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Normalised Importance (Integrated Gradients)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Legend for omics type
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=l)
                   for l, c in colour_map.items()]
        ax.legend(handles=handles, title='Omics', fontsize=8,
                  loc='lower right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved importance plot → {save_path}")
        return fig

    # ──────────────────────────────────────────────────────────────────────────
    # Save / load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, scores: np.ndarray, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as fh:
            pickle.dump(scores, fh)

    @staticmethod
    def load(path: str) -> np.ndarray:
        with open(path, 'rb') as fh:
            return pickle.load(fh)
