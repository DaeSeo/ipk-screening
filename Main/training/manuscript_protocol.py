"""Optional imbalanced ChemBERTa/XGBoost protocol.

The screenshot did not define its redundancy score or mixup pair selection.
This implementation makes both choices explicit: a negative's redundancy score
is its nearest other negative's cosine similarity; each mixup pair has a
positive anchor and a uniformly drawn training partner. Both steps run inside
the estimator's fit, so PyCaret cross-validation does not see generated data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedGroupKFold


def scaffold_groups(smiles: pd.Series) -> pd.Series:
    """Bemis–Murcko groups; acyclic molecules fall back to canonical SMILES."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    cache = {}
    groups = []
    for value in smiles:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("A valid standardized SMILES is needed for scaffold splitting")
        if value not in cache:
            mol = Chem.MolFromSmiles(value)
            if mol is None:
                raise ValueError(f"Cannot compute scaffold for {value!r}")
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            cache[value] = scaffold if scaffold else f"acyclic:{value}"
        groups.append(cache[value])
    return pd.Series(groups, index=smiles.index, name="scaffold")


def scaffold_holdout(labels, groups, *, test_size: float = 0.05, seed: int = 42):
    """Choose a scaffold-disjoint, approximately stratified test fold."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5")
    n_splits = round(1 / test_size)
    if n_splits > pd.Series(groups).nunique():
        raise ValueError("Not enough scaffolds for the requested test size")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates = []
    prevalence = float(np.mean(labels))
    for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels, groups):
        if labels.iloc[train_idx].nunique() < 2 or labels.iloc[test_idx].nunique() < 2:
            continue
        score = abs(len(test_idx) / len(labels) - test_size)
        score += abs(float(np.mean(labels.iloc[test_idx])) - prevalence)
        candidates.append((score, train_idx, test_idx))
    if not candidates:
        raise ValueError("No scaffold fold contains both hit and non-hit samples")
    _, train_idx, test_idx = min(candidates, key=lambda entry: entry[0])
    return train_idx, test_idx


def filter_redundant_negatives(X, y, *, offset: float = 0.1, chunk_size: int = 256):
    """Keep all positives and negatives with nearest-negative similarity <= μ−offset·σ.

    Cosine scores are computed in chunks to limit peak memory. This remains an
    O(number_of_negatives²) calculation; large datasets can take a long time.
    """
    if offset < 0 or chunk_size < 1:
        raise ValueError("offset must be nonnegative and chunk_size positive")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    neg_idx = np.flatnonzero(y == 0)
    if len(neg_idx) < 3:
        return np.arange(len(y)), float("nan")
    vectors = X[neg_idx]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    nearest = np.empty(len(neg_idx), dtype=np.float32)
    for start in range(0, len(neg_idx), chunk_size):
        stop = min(start + chunk_size, len(neg_idx))
        similarities = vectors[start:stop] @ vectors.T
        similarities[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        nearest[start:stop] = similarities.max(axis=1)
    threshold = float(nearest.mean() - offset * nearest.std())
    kept_neg = neg_idx[nearest <= threshold]
    if len(kept_neg) == 0:
        raise ValueError("Redundancy filtering removed all negative samples")
    keep = np.sort(np.r_[np.flatnonzero(y == 1), kept_neg])
    return keep, threshold


class RedundancyMixupXGB(ClassifierMixin, BaseEstimator):
    """A PyCaret-compatible classifier with fold-local filtering and soft mixup."""

    def __init__(
        self, redundancy_offset=0.1, mixup_alpha=0.4, mixup_multiplier=5,
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, n_jobs=1,
    ):
        self.redundancy_offset = redundancy_offset
        self.mixup_alpha = mixup_alpha
        self.mixup_multiplier = mixup_multiplier
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBRegressor

        matrix = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y, dtype=np.float32)
        if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
            raise ValueError("This estimator needs both binary classes in each training fold")
        if self.mixup_multiplier < 0 or not 0 < self.mixup_alpha < 1:
            raise ValueError("mixup_multiplier >= 0 and 0 < mixup_alpha < 1 are required")
        original_weights = np.ones(len(labels), dtype=np.float32)
        if sample_weight is not None:
            original_weights = np.asarray(sample_weight, dtype=np.float32)
            if len(original_weights) != len(labels) or (original_weights <= 0).any():
                raise ValueError("sample_weight must be positive and match training rows")

        kept, threshold = filter_redundant_negatives(
            matrix, labels, offset=self.redundancy_offset,
        )
        matrix, labels, weights = matrix[kept], labels[kept], original_weights[kept]
        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        class_weight = n_neg / n_pos
        weights[labels == 1] *= class_weight
        self.redundancy_threshold_ = threshold
        self.negative_rows_after_filter_ = n_neg
        self.positive_rows_ = n_pos
        self.positive_class_weight_ = class_weight

        n_aug = int(round(self.mixup_multiplier * n_pos))
        if n_aug:
            rng = np.random.default_rng(self.random_state)
            anchors = rng.choice(np.flatnonzero(labels == 1), size=n_aug, replace=True)
            partners = rng.integers(0, len(labels), size=n_aug)
            lam = rng.beta(self.mixup_alpha, self.mixup_alpha, size=n_aug).astype(np.float32)
            mixed_X = lam[:, None] * matrix[anchors] + (1 - lam[:, None]) * matrix[partners]
            mixed_y = lam * labels[anchors] + (1 - lam) * labels[partners]
            mixed_w = lam * weights[anchors] + (1 - lam) * weights[partners]
            matrix = np.concatenate([matrix, mixed_X])
            labels = np.concatenate([labels, mixed_y])
            weights = np.concatenate([weights, mixed_w])
        self.augmented_rows_ = n_aug
        # XGBClassifier validates integer class labels; XGBRegressor with the
        # logistic objective accepts targets in [0, 1] and outputs probabilities.
        self.model_ = XGBRegressor(
            objective="binary:logistic", n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, max_depth=self.max_depth,
            random_state=self.random_state, n_jobs=self.n_jobs,
            tree_method="hist", eval_metric="logloss",
        )
        self.model_.fit(matrix, labels, sample_weight=weights)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict_proba(self, X):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "model_")
        positive = np.clip(self.model_.predict(np.asarray(X, dtype=np.float32)), 0, 1)
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
