"""Shared inference and tabular I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Expected .csv or .parquet: {path}")


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Expected .csv or .parquet: {path}")
    return path


def choose_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def feature_frame(row_ids, values: np.ndarray, prefix: str) -> pd.DataFrame:
    if values.ndim != 2 or values.shape[0] != len(row_ids):
        raise ValueError("Embedding matrix must have one row per row_id")
    columns = [f"{prefix}_{i:04d}" for i in range(values.shape[1])]
    result = pd.DataFrame(values, columns=columns)
    result.insert(0, "row_id", list(row_ids))
    return result


def masked_mean(last_hidden_state, attention_mask):
    """Mean of nonpadding tokens; includes the model's special tokens."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
