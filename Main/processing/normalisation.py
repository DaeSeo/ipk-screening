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
    positive_if: str = "above",
    screening_target: str | None = None,
):
    """Add ``standardized_smiles`` and a binary ``activity`` column.

    For screening_target, both infection_percentage and cell_percentage must
    exceed 70 (hit70) or 50 (hit50); activity copies the selected hit column.
    Otherwise, activity compares one percentage with threshold inclusively.
    Missing/nonnumeric values receive a missing label.
    The input DataFrame and its original columns are left intact.
    """
    import pandas as pd

    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    if positive_if not in {"above", "below"}:
        raise ValueError("positive_if must be 'above' or 'below'")
    if screening_target not in {None, "hit70", "hit50"}:
        raise ValueError("screening_target must be 'hit70' or 'hit50'")

    measurement_columns = (("infection_percentage", "cell_percentage")
                           if screening_target else (activity_column,))
    missing = [name for name in (smiles_column, *measurement_columns) if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")

    result = df.copy()
    result["standardized_smiles"] = result[smiles_column].map(standardize_smiles)

    if screening_target:
        measurements = result[list(measurement_columns)].apply(pd.to_numeric, errors="coerce")
        invalid_range = measurements.notna() & (~measurements.ge(0) | ~measurements.le(100))
        if invalid_range.any().any():
            raise ValueError("infection_percentage and cell_percentage must be between 0 and 100")
        complete = measurements.notna().all(axis=1)
        for cutoff, label in ((70, "hit70"), (50, "hit50")):
            hit = measurements.gt(cutoff).all(axis=1)
            result[label] = hit.where(complete).astype("Int64")
        result["activity"] = result[screening_target].copy()
        return result

    percentages = pd.to_numeric(result[activity_column], errors="coerce")
    invalid_range = percentages.notna() & ~percentages.between(0, 100)
    if invalid_range.any():
        raise ValueError("infection percentages must be between 0 and 100")

    positive = percentages.ge(threshold) if positive_if == "above" else percentages.le(threshold)
    result["activity"] = positive.where(percentages.notna()).astype("Int64")
    return result
