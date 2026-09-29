"""Frozen ChemBERTa/MolFormer embeddings from standardised SMILES."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_extraction.common import choose_device, feature_frame, masked_mean
from feature_extraction.molecular import chemberta, molformer


MODELS = {"chemberta": chemberta, "molformer": molformer}


def extract_molecular_features(
    df: pd.DataFrame,
    *,
    model_name: str = "chemberta",
    batch_size: int = 32,
    max_length: int = 200,
    pooling: str = "mean",
) -> pd.DataFrame:
    """Encode each unique SMILES once; expand vectors back to original row IDs."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if not {"row_id", "standardized_smiles"}.issubset(df.columns):
        raise KeyError("Input needs row_id and standardized_smiles columns")
    if df["row_id"].isna().any() or df["row_id"].duplicated().any():
        raise ValueError("row_id must be unique and nonmissing")
    if batch_size < 1 or max_length < 2 or pooling not in {"mean", "pooler"}:
        raise ValueError("Check batch_size, max_length, and pooling")
    if model_name == "chemberta" and pooling == "pooler":
        raise ValueError("Use mean pooling for ChemBERTa; its regression checkpoint has no trained pooler")

    valid = df.loc[df["standardized_smiles"].notna(), ["row_id", "standardized_smiles"]].copy()
    valid = valid.loc[valid["standardized_smiles"].astype(str).str.strip().ne("")]
    if valid.empty:
        raise ValueError("No valid standardized_smiles to embed")
    unique_smiles = valid["standardized_smiles"].drop_duplicates().tolist()

    spec = MODELS[model_name]
    kwargs = {"trust_remote_code": spec.TRUST_REMOTE_CODE}
    if spec.REVISION is not None:
        kwargs["revision"] = spec.REVISION
    tokenizer = AutoTokenizer.from_pretrained(spec.MODEL_ID, **kwargs)
    model = AutoModel.from_pretrained(spec.MODEL_ID, **kwargs)
    device = choose_device()
    model.to(device).eval()

    vectors = []
    with torch.inference_mode():
        for start in range(0, len(unique_smiles), batch_size):
            batch = unique_smiles[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            if pooling == "pooler":
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    raise ValueError(f"{model_name} has no pooler_output; use --pooling mean")
            else:
                pooled = masked_mean(outputs.last_hidden_state, inputs["attention_mask"])
            vectors.append(pooled.cpu().float().numpy())

    embedding_matrix = np.concatenate(vectors).astype(np.float32)
    unique = feature_frame(unique_smiles, embedding_matrix, spec.PREFIX)
    unique = unique.rename(columns={"row_id": "standardized_smiles"})
    result = valid.merge(unique, on="standardized_smiles", how="left", validate="many_to_one")
    return result.drop(columns="standardized_smiles")
