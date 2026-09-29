"""Explicit held-out binary metrics for the biological hit class."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_holdout(predictions: pd.DataFrame, *, hit_label: int = 1) -> pd.DataFrame:
    """Compute metrics using P(hit), rather than PyCaret's predicted-class score.

    PR-AUC here is sklearn average precision (non-interpolated), which differs
    from trapezoidal area under a precision-recall curve.
    """
    if hit_label not in (0, 1):
        raise ValueError("hit_label must be 0 or 1")
    score_col = f"prediction_score_{hit_label}"
    needed = {"activity", "prediction_label", score_col}
    missing = needed - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}. "
            "Generate holdout predictions with raw_score=True."
        )
    y_true = pd.to_numeric(predictions["activity"], errors="raise").to_numpy()
    y_pred = pd.to_numeric(predictions["prediction_label"], errors="raise").to_numpy()
    scores = pd.to_numeric(predictions[score_col], errors="raise").to_numpy(dtype=float)
    if (not np.isin(y_true, [0, 1]).all()
            or not np.isin(y_pred, [0, 1]).all()
            or len(np.unique(y_true)) != 2):
        raise ValueError("Holdout must contain both 0/1 classes and binary predictions")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Hit probability must be finite and between 0 and 1")
    hit_true = (y_true == hit_label).astype(int)
    hit_pred = (y_pred == hit_label).astype(int)
    tn, fp, fn, tp = confusion_matrix(hit_true, hit_pred, labels=[0, 1]).ravel()
    return pd.DataFrame([{
        "hit_label": hit_label,
        "test_rows": len(y_true),
        "test_hits": int(hit_true.sum()),
        "hit_prevalence": float(hit_true.mean()),
        "F1": float(f1_score(hit_true, hit_pred, zero_division=0)),
        "Precision": float(precision_score(hit_true, hit_pred, zero_division=0)),
        "Recall": float(recall_score(hit_true, hit_pred, zero_division=0)),
        "Accuracy": float(accuracy_score(hit_true, hit_pred)),
        "PR-AUC (AP)": float(average_precision_score(hit_true, scores)),
        "ROC-AUC": float(roc_auc_score(hit_true, scores)),
        "Balanced Accuracy": float(balanced_accuracy_score(hit_true, hit_pred)),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }])
