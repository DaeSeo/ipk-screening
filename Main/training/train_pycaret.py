"""Group-aware PyCaret classification on assembled embedding tables."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from feature_extraction.common import read_table


FEATURE_PREFIXES = ("cb_", "mf_", "img_")


def prepare_training_data(df: pd.DataFrame, group_column: str = "standardized_smiles"):
    """Return numeric features, binary labels, group keys, and original row IDs."""
    if "row_id" not in df or "activity" not in df:
        raise KeyError("Input needs row_id and activity columns")
    feature_cols = [column for column in df if column.startswith(FEATURE_PREFIXES)]
    if not feature_cols:
        raise ValueError("No cb_, mf_, or img_ embedding columns found")
    if df["row_id"].isna().any() or df["row_id"].duplicated().any():
        raise ValueError("row_id must be unique and nonmissing")

    matrix = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    labels = pd.to_numeric(df["activity"], errors="coerce")
    keep = labels.isin([0, 1]) & matrix.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    subset = df.loc[keep].reset_index(drop=True)
    matrix = matrix.loc[keep].reset_index(drop=True).astype("float32")
    labels = labels.loc[keep].reset_index(drop=True).astype(int)
    if subset.empty or labels.nunique() != 2:
        raise ValueError("Need complete embeddings and both activity classes (0 and 1)")

    if group_column in subset.columns:
        group_values = subset[group_column].astype("string").str.strip()
        group_values = group_values.mask(group_values.eq(""))
        groups = group_values.fillna("row:" + subset["row_id"].astype(str))
    elif group_column == "standardized_smiles":
        groups = "row:" + subset["row_id"].astype(str)
    else:
        raise KeyError(f"Missing requested group column: {group_column}")
    return matrix, labels, groups.reset_index(drop=True), subset["row_id"]


def group_holdout(labels, groups, *, test_size: float = 0.2, seed: int = 42):
    """Select a group-disjoint split containing both classes on both sides."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if pd.Series(groups).nunique() < 3:
        raise ValueError("At least three distinct compound groups are needed")
    splitter = GroupShuffleSplit(n_splits=50, test_size=test_size, random_state=seed)
    for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels, groups):
        if labels.iloc[train_idx].nunique() == labels.iloc[test_idx].nunique() == 2:
            return train_idx, test_idx
    raise ValueError("Cannot find a group-disjoint split containing both classes")


def train_classifier(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str | None = None,
    group_column: str = "standardized_smiles",
    test_size: float | None = None,
    folds: int = 5,
    seed: int = 42,
    jobs: int = 1,
    protocol: str = "baseline",
    supplementary_column: str | None = None,
    supplementary_boost: float = 1.0,
    redundancy_offset: float = 0.1,
    mixup_alpha: float = 0.4,
    mixup_multiplier: float = 5.0,
) -> Path:
    """Train with PyCaret, evaluate a held-out group split, and save artifacts."""
    if jobs < 1 or folds < 2:
        raise ValueError("jobs must be >= 1 and folds >= 2")
    if protocol not in {"baseline", "scaffold", "manuscript"}:
        raise ValueError("protocol must be baseline, scaffold, or manuscript")
    if supplementary_boost < 1:
        raise ValueError("supplementary_boost must be at least 1")
    if test_size is None:
        test_size = 0.05 if protocol in {"scaffold", "manuscript"} else 0.2
    if model_name is None:
        model_name = "xgboost" if protocol in {"scaffold", "manuscript"} else "rf"
    if protocol in {"scaffold", "manuscript"} and model_name != "xgboost":
        raise ValueError("Scaffold and manuscript protocols use --model xgboost")
    # Limits nested native thread pools on Mac, where unrestricted XGBoost can
    # exhaust memory or crash a notebook kernel.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(jobs)
    from pycaret.classification import ClassificationExperiment

    data = read_table(input_path)
    features, labels, groups, row_ids = prepare_training_data(data, group_column)
    if protocol in {"scaffold", "manuscript"}:
        from training.manuscript_protocol import scaffold_groups, scaffold_holdout

        if not all(name.startswith("cb_") for name in features.columns):
            raise ValueError("Scaffold and manuscript modes expect ChemBERTa features only")
        if "standardized_smiles" not in data:
            raise KeyError("Scaffold and manuscript modes need standardized_smiles")
        smiles = data.set_index("row_id").loc[row_ids, "standardized_smiles"].reset_index(drop=True)
        groups = scaffold_groups(smiles)
        train_idx, test_idx = scaffold_holdout(labels, groups, test_size=test_size, seed=seed)
    else:
        train_idx, test_idx = group_holdout(labels, groups, test_size=test_size, seed=seed)
    n_groups = groups.iloc[train_idx].nunique()
    cv_folds = min(folds, n_groups)
    if cv_folds < 2:
        raise ValueError("Training split has fewer than two compound groups")

    target = "activity"
    train = features.iloc[train_idx].reset_index(drop=True)
    train[target] = labels.iloc[train_idx].reset_index(drop=True)
    holdout = features.iloc[test_idx].reset_index(drop=True)
    holdout[target] = labels.iloc[test_idx].reset_index(drop=True)
    train_groups = groups.iloc[train_idx].reset_index(drop=True)
    train_weights = None
    if supplementary_column:
        if supplementary_column not in data:
            raise KeyError(f"Missing supplementary column: {supplementary_column}")
        flags = data.set_index("row_id").loc[row_ids, supplementary_column]
        flags = pd.to_numeric(flags, errors="raise").reset_index(drop=True)
        if not flags.isin([0, 1]).all():
            raise ValueError("supplementary column must contain only 0 and 1")
        train_weights = (1 + (supplementary_boost - 1) * flags.iloc[train_idx]).reset_index(drop=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_id": row_ids.iloc[train_idx].to_numpy(),
        "group": train_groups.to_numpy(), "activity": train[target].to_numpy(),
        "split": "train",
    }).to_csv(output_dir / "train_rows.csv", index=False)
    pd.DataFrame({
        "row_id": row_ids.iloc[test_idx].to_numpy(),
        "group": groups.iloc[test_idx].to_numpy(), "activity": holdout[target].to_numpy(),
        "split": "test",
    }).to_csv(output_dir / "test_rows.csv", index=False)

    if protocol in {"scaffold", "manuscript"}:
        from sklearn.model_selection import StratifiedGroupKFold

        fold_strategy = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    else:
        fold_strategy = "groupkfold"
    exp = ClassificationExperiment()
    exp.setup(
        data=train, target=target, test_data=holdout, index=False, session_id=seed,
        fold_strategy=fold_strategy, fold_groups=train_groups, fold=cv_folds,
        n_jobs=jobs, use_gpu=False, verbose=False, html=False,
    )
    if protocol == "manuscript":
        from training.manuscript_protocol import RedundancyMixupXGB

        estimator = RedundancyMixupXGB(
            redundancy_offset=redundancy_offset, mixup_alpha=mixup_alpha,
            mixup_multiplier=mixup_multiplier, random_state=seed, n_jobs=jobs,
        )
        fit_kwargs = {"sample_weight": train_weights.to_numpy()} if train_weights is not None else {}
        best = exp.create_model(estimator, fit_kwargs=fit_kwargs, verbose=False)
    else:
        available = set(exp.models().index)
        if model_name == "compare":
            candidates = [item for item in ("lr", "rf", "et", "xgboost") if item in available]
            if not candidates:
                raise RuntimeError("PyCaret found no supported candidate models")
            best = exp.compare_models(include=candidates, sort="AUC", verbose=False)
        else:
            if model_name not in available:
                raise ValueError(
                    f"PyCaret model {model_name!r} is unavailable. "
                    f"Available examples: {sorted(available)[:20]}. "
                    "For xgboost, install the xgboost package first."
                )
            best = exp.create_model(model_name, verbose=False)
    exp.pull().to_csv(output_dir / "cross_validation.csv", index=False)
    exp.predict_model(best, verbose=False)
    exp.pull().to_csv(output_dir / "holdout_metrics.csv", index=False)
    predictions = exp.predict_model(best, data=holdout.drop(columns=target), verbose=False)
    predictions = predictions.reset_index(drop=True)
    if len(predictions) != len(holdout):
        raise RuntimeError("PyCaret returned a different number of holdout predictions")
    predictions.insert(0, "row_id", row_ids.iloc[test_idx].to_numpy())
    predictions.insert(1, "activity", holdout[target].to_numpy())
    predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
    exp.save_model(best, str(output_dir / "model"), verbose=False)
    metadata = {
        "model_requested": model_name, "protocol": protocol,
        "group_column": "Bemis-Murcko scaffold" if protocol in {"scaffold", "manuscript"} else group_column,
        "train_rows": len(train), "test_rows": len(holdout),
        "features": list(features.columns), "folds": int(cv_folds),
        "seed": seed, "jobs": jobs, "test_size": test_size,
        "train_hits": int(train[target].sum()), "test_hits": int(holdout[target].sum()),
        "supplementary_column": supplementary_column, "supplementary_boost": supplementary_boost,
        "redundancy_offset": redundancy_offset if protocol == "manuscript" else None,
        "mixup_alpha": mixup_alpha if protocol == "manuscript" else None,
        "mixup_multiplier": mixup_multiplier if protocol == "manuscript" else None,
    }
    if protocol == "manuscript":
        metadata["final_fit"] = {
            name: getattr(best, attr, None)
            for name, attr in (
                ("non_hits_after_filter", "negative_rows_after_filter_"),
                ("hits", "positive_rows_"),
                ("positive_class_weight", "positive_class_weight_"),
                ("augmented_rows", "augmented_rows_"),
                ("redundancy_threshold", "redundancy_threshold_"),
            )
        }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2))
    return output_dir
