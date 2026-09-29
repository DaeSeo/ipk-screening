"""DINOv2 embeddings from an image manifest, aggregated per dataset row."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from feature_extraction.common import choose_device, feature_frame


MODEL_ID = "facebook/dinov2-base"


def extract_image_features(
    manifest: pd.DataFrame,
    *,
    image_root: str | Path,
    batch_size: int = 16,
    model_id: str = MODEL_ID,
) -> pd.DataFrame:
    """Compute CLS embeddings and mean-pool multiple images for each row_id.

    Manifest columns: row_id, image_path. Paths may be absolute or relative to
    image_root. Inputs are loaded as RGB; inspect microscopy channels first.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    if not {"row_id", "image_path"}.issubset(manifest.columns):
        raise KeyError("Image manifest needs row_id and image_path columns")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = manifest[["row_id", "image_path"]].dropna().copy()
    if rows.empty:
        raise ValueError("Image manifest has no complete rows")
    rows["row_id"] = pd.to_numeric(rows["row_id"], errors="raise").astype("int64")
    if rows["image_path"].astype(str).str.strip().eq("").any():
        raise ValueError("image_path cannot be empty")
    root = Path(image_root)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    device = choose_device()
    model.to(device).eval()

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = defaultdict(int)
    records = list(rows.itertuples(index=False, name=None))
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            images = []
            for _, image_path in batch:
                path = Path(image_path)
                if not path.is_absolute():
                    path = root / path
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            vectors = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
            for (row_id, _), vector in zip(batch, vectors):
                row_id = int(row_id)
                if row_id not in sums:
                    sums[row_id] = np.zeros_like(vector, dtype=np.float64)
                sums[row_id] += vector
                counts[row_id] += 1

    ids = sorted(sums)
    values = np.stack([(sums[i] / counts[i]).astype(np.float32) for i in ids])
    return feature_frame(ids, values, "img")
