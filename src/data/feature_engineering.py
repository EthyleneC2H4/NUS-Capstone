"""
Feature engineering pipeline for multi-omics node features.

Improvements over raw features used in the benchmark:
- Z-score standardisation: centres each feature, unit variance.
- Min-max normalisation: maps features to [0, 1].
- Variance-based feature selection: removes near-constant features.
- Optional PCA: reduces dimensionality while preserving most variance.

All transformers are fitted on training-set nodes only and then applied
to the full node matrix to prevent data leakage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureEngineer:
    """
    Sklearn-style pipeline: (optional select) → (optional scale) → (optional PCA).

    Parameters
    ----------
    normalize : str | None
        'standard' (Z-score), 'minmax', or None.
    feature_selection : bool
        Whether to drop near-zero-variance features.
    var_threshold : float
        Minimum variance to keep a feature (only when feature_selection=True).
    use_pca : bool
        Whether to apply PCA after scaling.
    pca_components : int | float | None
        Number of components (int), fraction of variance to keep (float 0-1),
        or None (keep all non-trivial components).
    """

    def __init__(
        self,
        normalize: Optional[str] = 'standard',
        feature_selection: bool = True,
        var_threshold: float = 0.01,
        use_pca: bool = False,
        pca_components: Optional[int | float] = None,
    ):
        self.normalize = normalize
        self.feature_selection = feature_selection
        self.var_threshold = var_threshold
        self.use_pca = use_pca
        self.pca_components = pca_components

        self._selector: Optional[VarianceThreshold] = None
        self._scaler: Optional[StandardScaler | MinMaxScaler] = None
        self._pca: Optional[PCA] = None
        self.fitted = False

    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, features: np.ndarray) -> 'FeatureEngineer':
        """Fit all pipeline stages on *features* (numpy array, shape N×D)."""

        if self.feature_selection:
            self._selector = VarianceThreshold(threshold=self.var_threshold)
            features = self._selector.fit_transform(features)

        if self.normalize == 'standard':
            self._scaler = StandardScaler()
            features = self._scaler.fit_transform(features)
        elif self.normalize == 'minmax':
            self._scaler = MinMaxScaler()
            features = self._scaler.fit_transform(features)

        if self.use_pca:
            self._pca = PCA(n_components=self.pca_components, random_state=42)
            self._pca.fit(features)

        self.fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted pipeline to *features*."""
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")

        if self._selector is not None:
            features = self._selector.transform(features)
        if self._scaler is not None:
            features = self._scaler.transform(features)
        if self._pca is not None:
            features = self._pca.transform(features)
        return features

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    # ──────────────────────────────────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        """Feature dimensionality produced by this pipeline."""
        if self._pca is not None:
            return self._pca.n_components_
        if self._selector is not None:
            return int(self._selector.get_support().sum())
        return None   # unchanged

    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Return PCA explained variance ratio (None if PCA not used)."""
        if self._pca is not None:
            return self._pca.explained_variance_ratio_
        return None

    def selected_feature_mask(self) -> Optional[np.ndarray]:
        """Boolean mask of kept features (None if selection not used)."""
        if self._selector is not None:
            return self._selector.get_support()
        return None
