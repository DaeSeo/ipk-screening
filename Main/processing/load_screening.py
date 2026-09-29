"""Load and align the 2024 Excel and 2025 CSV compound screens."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INFECTION_COLUMN = "infected cells / number of cells (%)"
CELL_COLUMN = "cellratio(%)"


def convert_well_format(well: object) -> object:
    """Convert a well such as A01 to A1; preserve missing/other values."""
    if isinstance(well, str) and len(well) == 3 and well[1] == "0":
        return well[0] + well[2]
    return well


def _prepare_screen(df: pd.DataFrame, year: int, id_column: str) -> pd.DataFrame:
    required = ["SMILES", "Barcode", "Well", INFECTION_COLUMN, CELL_COLUMN]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise KeyError(f"{year} input is missing column(s): {', '.join(missing)}")

    result = df.rename(columns={
        "SMILES": "smiles",
        "Barcode": "plate_name",
        "Well": "well",
        INFECTION_COLUMN: "infection_percentage",
        CELL_COLUMN: "cell_percentage",
        id_column: "compound_id",
    }).copy()
    smiles = result["smiles"].astype("string").str.strip()
    result = result.loc[smiles.notna() & smiles.ne("") & smiles.ne("--")].copy()
    result["smiles"] = smiles.loc[result.index]
    result["well"] = result["well"].map(convert_well_format)
    result["screening_year"] = str(year)
    result["year"] = year
    columns = ["smiles", "infection_percentage", "cell_percentage",
               "screening_year", "plate_name", "well", "year"]
    if "compound_id" in result:
        columns.append("compound_id")
    return result[columns]


def load_screening_data(input_2024: str | Path, input_2025: str | Path) -> pd.DataFrame:
    """Return one aligned table, with 2024 rows followed by 2025 rows."""
    data_2024 = pd.read_excel(input_2024)
    data_2025 = pd.read_csv(input_2025)
    old = _prepare_screen(data_2024, 2024, "Compound ID")
    new = _prepare_screen(data_2025, 2025, "Compound_ID")
    return pd.concat([old, new], ignore_index=True)
