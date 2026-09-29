"""SMILES standardisation and activity labels for the input dataset."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


DEFAULT_ACTIVITY_COLUMN = "infected cells / number of cells (%)"


def standardize_smiles(smiles: str | None) -> str | None:
    """Clean a SMILES, keep its largest fragment, and return canonical SMILES.

    Stereochemistry is retained. Invalid or missing inputs return None.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None

    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol)
    result = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return result or None


def normalise_dataframe(
    df,
    *,
    smiles_column: str = "SMILES",
    activity_column: str = DEFAULT_ACTIVITY_COLUMN,
    threshold: float = 70,
    positive_if: str = "below",
):
    """Add ``standardized_smiles`` and a binary ``activity`` column.

    ``activity`` is 1 when the percentage is >= threshold (``above``) or
    <= threshold (``below``). Missing/nonnumeric values receive a missing label.
    The input DataFrame and its original columns are left intact.
    """
    import pandas as pd

    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    if positive_if not in {"above", "below"}:
        raise ValueError("positive_if must be 'above' or 'below'")

    missing = [name for name in (smiles_column, activity_column) if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")

    result = df.copy()
    result["standardized_smiles"] = result[smiles_column].map(standardize_smiles)

    percentages = pd.to_numeric(result[activity_column], errors="coerce")
    invalid_range = percentages.notna() & ~percentages.between(0, 100)
    if invalid_range.any():
        raise ValueError("infection percentages must be between 0 and 100")

    positive = percentages.ge(threshold) if positive_if == "above" else percentages.le(threshold)
    result["activity"] = positive.where(percentages.notna()).astype("Int64")
    return result
