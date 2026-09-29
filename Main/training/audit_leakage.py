"""Audit a saved experiment for overlap, label alignment, and class imbalance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

from feature_extraction.common import read_table


def audit_run(features_path: str | Path, run_dir: str | Path) -> dict:
    """Inspect saved train/test IDs against features and holdout predictions."""
    run_dir = Path(run_dir)
    data = read_table(features_path)
    train = pd.read_csv(run_dir / "train_rows.csv")
    test = pd.read_csv(run_dir / "test_rows.csv")
    predictions = pd.read_csv(run_dir / "holdout_predictions.csv")
    required = {"row_id", "activity", "standardized_smiles"}
    if not required.issubset(data.columns):
        raise KeyError(f"Feature table is missing {sorted(required - set(data.columns))}")
    for name, frame, columns in (
        ("features", data, required),
        ("train_rows", train, {"row_id", "activity", "group"}),
        ("test_rows", test, {"row_id", "activity", "group"}),
        ("holdout_predictions", predictions, {"row_id", "activity", "prediction_label"}),
    ):
        missing = columns - set(frame.columns)
        if missing:
            raise KeyError(f"{name} is missing {sorted(missing)}")
        if frame.row_id.isna().any() or frame.row_id.duplicated().any():
            raise ValueError(f"{name} needs unique, nonmissing row_id")

    ids_train, ids_test = set(train.row_id), set(test.row_id)
    group_train = set(train.group.dropna().astype(str))
    group_test = set(test.group.dropna().astype(str))
    indexed = data.set_index("row_id")
    train_missing = ids_train - set(indexed.index)
    test_missing = ids_test - set(indexed.index)
    if train_missing or test_missing:
        raise ValueError("Saved train/test row_id is absent from the supplied feature table")

    train_smiles = set(indexed.loc[train.row_id, "standardized_smiles"].dropna())
    test_smiles = set(indexed.loc[test.row_id, "standardized_smiles"].dropna())
    feature_cols = [c for c in data if c.startswith(("cb_", "mf_", "img_"))]
    if not feature_cols:
        raise ValueError("No embedding columns found")
    train_hash = pd.util.hash_pandas_object(indexed.loc[train.row_id, feature_cols], index=False)
    test_hash = pd.util.hash_pandas_object(indexed.loc[test.row_id, feature_cols], index=False)

    test_labels = pd.to_numeric(test.activity, errors="raise")
    feature_labels = pd.to_numeric(indexed.loc[test.row_id, "activity"], errors="raise")
    train_labels = pd.to_numeric(indexed.loc[train.row_id, "activity"], errors="raise")
    aligned_predictions = predictions.set_index("row_id").reindex(test.row_id)
    prediction_labels = pd.to_numeric(aligned_predictions.prediction_label, errors="raise")
    predicted_truth = pd.to_numeric(aligned_predictions.activity, errors="raise")
    if aligned_predictions.prediction_label.isna().any() or len(predictions) != len(test):
        raise ValueError("Predictions do not match the saved test row IDs")
    true = test_labels.to_numpy(dtype=int)
    pred = prediction_labels.to_numpy(dtype=int)
    if not np.isin(true, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("This audit expects binary 0/1 labels and predictions")
    majority_label = int(pd.Series(true).value_counts().idxmax())
    majority = np.full_like(true, majority_label)
    result = {
        "train_rows": len(train), "test_rows": len(test),
        "train_positive_fraction": float(np.mean(train_labels.to_numpy(dtype=int) == 1)),
        "test_positive_fraction": float(np.mean(true == 1)),
        "predicted_positive_fraction": float(np.mean(pred == 1)),
        "train_test_row_id_overlap": len(ids_train & ids_test),
        "train_test_group_overlap": len(group_train & group_test),
        "train_test_standardized_smiles_overlap": len(train_smiles & test_smiles),
        "train_test_identical_embedding_hashes": len(set(train_hash) & set(test_hash)),
        "train_label_mismatches": int(np.count_nonzero(train_labels.to_numpy() != train.activity.to_numpy())),
        "test_label_mismatches": int(np.count_nonzero(feature_labels.to_numpy() != true)),
        "prediction_truth_mismatches": int(np.count_nonzero(predicted_truth.to_numpy() != true)),
        "f1_class_1": float(f1_score(true, pred, pos_label=1, zero_division=0)),
        "f1_class_0": float(f1_score(true, pred, pos_label=0, zero_division=0)),
        "precision_class_1": float(precision_score(true, pred, pos_label=1, zero_division=0)),
        "recall_class_1": float(recall_score(true, pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "majority_label": majority_label,
        "majority_f1_class_1": float(f1_score(true, majority, pos_label=1, zero_division=0)),
        "majority_balanced_accuracy": float(balanced_accuracy_score(true, majority)),
    }
    return result


def print_audit(features_path: str | Path, run_dir: str | Path) -> dict:
    result = audit_run(features_path, run_dir)
    report = Path(run_dir) / "leakage_audit.json"
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved audit to {report}")
    return result
