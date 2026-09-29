"""Join selected embeddings to labels by stable row_id."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_extraction.common import read_table, save_table


def assemble_features(
    processed: str | Path,
    embedding_files: list[str | Path],
    output: str | Path,
    *,
    compound_column: str | None = None,
    supplementary_column: str | None = None,
) -> pd.DataFrame:
    """Keep only rows for which all selected modalities have embeddings."""
    if not embedding_files:
        raise ValueError("Choose at least one embedding file")
    df = read_table(processed)
    required = {"row_id", "activity"}
    if not required.issubset(df.columns):
        raise KeyError(f"Processed data must contain {sorted(required)}")
    if df["row_id"].duplicated().any() or df["row_id"].isna().any():
        raise ValueError("Processed row_id must be unique and nonmissing")
    columns = ["row_id", "activity"]
    if "standardized_smiles" in df:
        columns.append("standardized_smiles")
    for optional_column in (compound_column, supplementary_column):
        if optional_column:
            if optional_column not in df.columns:
                raise KeyError(f"Missing source column: {optional_column}")
            if optional_column not in columns:
                columns.append(optional_column)
    result = df[columns].copy()
    feature_columns: set[str] = set()

    for path in embedding_files:
        features = read_table(path)
        if "row_id" not in features or features["row_id"].duplicated().any():
            raise ValueError(f"Embedding file needs unique row_id: {path}")
        new = [name for name in features if name != "row_id"]
        if not new or not all(name.startswith(("cb_", "mf_", "img_")) for name in new):
            raise ValueError(f"Unrecognised embedding columns in {path}")
        if feature_columns.intersection(new):
            raise ValueError(f"Duplicate feature columns in {path}")
        feature_columns.update(new)
        result = result.merge(features, on="row_id", how="inner", validate="one_to_one")

    if result.empty:
        raise ValueError("No overlapping row_id values across selected files")
    save_table(result, output)
    return result
