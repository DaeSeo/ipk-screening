"""CSV preprocessing, embedding extraction, feature joining, and training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def locate_data_dir() -> Path:
    """Prefer the Data folder containing a source CSV; support either layout."""
    script_dir = Path(__file__).resolve().parent
    choices = (script_dir / "Data", script_dir.parent / "Data")
    for directory in choices:
        if directory.is_dir() and any(
            path.is_file() and path.suffix.lower() == ".csv"
            and path.name.lower() not in {"processed.csv", "invalid_smiles.csv"}
            for path in directory.iterdir()
        ):
            return directory
    return next((directory for directory in choices if directory.is_dir()), choices[1])


DATA_DIR = locate_data_dir()
DEFAULT_ACTIVITY_COLUMN = "infected cells / number of cells (%)"
DEFAULT_OUTPUT = DATA_DIR / "processed.csv"


def find_input_csv(data_dir: str | Path = DATA_DIR) -> Path:
    """Find one raw CSV, excluding the generated processed.csv."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Data folder does not exist: {directory}")
    candidates = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
        and path.name.lower() not in {"processed.csv", "invalid_smiles.csv"}
    )
    if not candidates:
        raise FileNotFoundError(f"No input CSV found in {directory}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Multiple input CSV files found ({names}); choose one with --input")
    return candidates[0]


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def process_data(
    df: pd.DataFrame,
    *,
    threshold: float = 70,
    smiles_column: str = "SMILES",
    activity_column: str = DEFAULT_ACTIVITY_COLUMN,
    positive_if: str = "below",
) -> pd.DataFrame:
    """Normalise SMILES, label activity, and retain a stable source row ID."""
    from processing.normalisation import normalise_dataframe

    source = df.copy()
    if "row_id" not in source:
        source.insert(0, "row_id", range(len(source)))
    if source["row_id"].isna().any() or source["row_id"].duplicated().any():
        raise ValueError("row_id must be unique and nonmissing")
    return normalise_dataframe(
        source, threshold=threshold, smiles_column=smiles_column,
        activity_column=activity_column, positive_if=positive_if,
    )


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_pipeline(
    input_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    threshold: float = 70,
    smiles_column: str = "SMILES",
    activity_column: str = DEFAULT_ACTIVITY_COLUMN,
    positive_if: str = "below",
) -> pd.DataFrame:
    """Read, normalise, label, save, and return the processed CSV."""
    df = read_csv(find_input_csv() if input_path is None else input_path)
    result = process_data(
        df, threshold=threshold, smiles_column=smiles_column,
        activity_column=activity_column, positive_if=positive_if,
    )
    save_csv(result, output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preprocess = commands.add_parser("preprocess", help="SMILES and activity labels")
    preprocess.add_argument("--input", help="Raw CSV; defaults to the sole raw CSV in Data")
    preprocess.add_argument("--output", default=DEFAULT_OUTPUT, help="Default: Data/processed.csv")
    preprocess.add_argument("--invalid-output", help="Default: invalid_smiles.csv beside processed.csv")
    preprocess.add_argument("--threshold", type=float, default=70)
    preprocess.add_argument("--smiles-column", default="SMILES")
    preprocess.add_argument("--activity-column", default=DEFAULT_ACTIVITY_COLUMN)
    preprocess.add_argument("--positive-if", choices=["above", "below"], default="below",
                            help="Label 1 for percentages >= or <= threshold")

    molecular = commands.add_parser("embed-molecular", help="SMILES embeddings")
    molecular.add_argument("--input", default=DEFAULT_OUTPUT)
    molecular.add_argument("--model", choices=["chemberta", "molformer"], default="chemberta")
    molecular.add_argument("--output", help="Default: Data/embeddings/molecular/MODEL.parquet")
    molecular.add_argument("--batch-size", type=int, default=32)
    molecular.add_argument("--max-length", type=int, default=200)
    molecular.add_argument("--pooling", choices=["mean", "pooler"], default="mean")

    images = commands.add_parser("embed-images", help="DINOv2 image embeddings")
    images.add_argument("--manifest", default=DATA_DIR / "manifests" / "images.csv")
    images.add_argument("--image-root", default=DATA_DIR / "images")
    images.add_argument("--output", default=DATA_DIR / "embeddings" / "image" / "dinov2.parquet")
    images.add_argument("--batch-size", type=int, default=16)
    images.add_argument("--model-id", default="facebook/dinov2-base")

    assemble = commands.add_parser("assemble", help="Join features to labels on row_id")
    assemble.add_argument("--input", default=DEFAULT_OUTPUT)
    assemble.add_argument("--molecular", choices=["chemberta", "molformer"], nargs="*", default=[])
    assemble.add_argument("--images", action="store_true", help="Include DINOv2 image features")
    assemble.add_argument("--compound-column", help="Optional source compound ID to preserve")
    assemble.add_argument("--supplementary-column", help="Optional 0/1 flag for extra screening data")
    assemble.add_argument("--output", help="Default: Data/features/selected modalities.parquet")

    train = commands.add_parser("train", help="Group-aware PyCaret classification")
    train.add_argument("--input", required=True, help="Assembled .parquet or .csv")
    train.add_argument("--output-dir", default=DATA_DIR / "models" / "experiment")
    train.add_argument("--model", help="PyCaret model ID; default rf (baseline), xgboost (scaffold), or custom XGBoost (manuscript)")
    train.add_argument("--protocol", choices=["baseline", "scaffold", "manuscript"], default="baseline")
    train.add_argument("--group-column", default="standardized_smiles",
                       help="Group identical compounds across holdout/CV")
    train.add_argument("--test-size", type=float, help="Default 0.2 (baseline), 0.05 (scaffold/manuscript)")
    train.add_argument("--supplementary-column", help="0/1 flag for compounds with supplementary screening")
    train.add_argument("--supplementary-boost", type=float, default=1.0)
    train.add_argument("--redundancy-offset", type=float, default=0.1)
    train.add_argument("--mixup-alpha", type=float, default=0.4)
    train.add_argument("--mixup-multiplier", type=float, default=5.0)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--jobs", type=int, default=1, help="Default 1 avoids Mac oversubscription")

    audit = commands.add_parser("audit", help="Check saved split, predictions, and class balance")
    audit.add_argument("--input", required=True, help="Feature table used for training")
    audit.add_argument("--run-dir", required=True, help="Directory with saved training results")
    return parser


def main(argv: list[str] | None = None) -> None:
    args_in = list(sys.argv[1:] if argv is None else argv)
    # Keep the original `python main.py --input ...` invocation working.
    if not args_in or (args_in[0].startswith("--") and args_in[0] != "--help"):
        args_in.insert(0, "preprocess")
    args = build_parser().parse_args(args_in)

    if args.command == "preprocess":
        result = run_pipeline(
            args.input, args.output, threshold=args.threshold,
            smiles_column=args.smiles_column,
            activity_column=args.activity_column, positive_if=args.positive_if,
        )
        invalid_output = args.invalid_output or Path(args.output).with_name("invalid_smiles.csv")
        save_csv(result.loc[result["standardized_smiles"].isna()], invalid_output)
        labels = result["activity"].value_counts(dropna=False).to_dict()
        print(f"Saved {len(result)} rows to {args.output}; "
              f"invalid SMILES: {result['standardized_smiles'].isna().sum()} "
              f"(see {invalid_output}); activity counts: {labels}")
    elif args.command == "embed-molecular":
        from feature_extraction.common import read_table, save_table
        from feature_extraction.molecular.extract import extract_molecular_features

        output = args.output or DATA_DIR / "embeddings" / "molecular" / f"{args.model}.parquet"
        result = extract_molecular_features(
            read_table(args.input), model_name=args.model,
            batch_size=args.batch_size, max_length=args.max_length, pooling=args.pooling,
        )
        save_table(result, output)
        print(f"Saved {len(result)} molecular embeddings to {output}")
    elif args.command == "embed-images":
        from feature_extraction.common import read_table, save_table
        from feature_extraction.image.dino import extract_image_features

        result = extract_image_features(
            read_table(args.manifest), image_root=args.image_root,
            batch_size=args.batch_size, model_id=args.model_id,
        )
        save_table(result, args.output)
        print(f"Saved {len(result)} aggregated image embeddings to {args.output}")
    elif args.command == "assemble":
        from feature_extraction.assemble import assemble_features

        selected = [DATA_DIR / "embeddings" / "molecular" / f"{name}.parquet"
                    for name in args.molecular]
        if args.images:
            selected.append(DATA_DIR / "embeddings" / "image" / "dinov2.parquet")
        if not selected:
            raise ValueError("Choose --molecular chemberta/molformer and/or --images")
        modality_name = "_".join([*args.molecular, *(["dinov2"] if args.images else [])])
        output = args.output or DATA_DIR / "features" / f"{modality_name}.parquet"
        result = assemble_features(
            args.input, selected, output, compound_column=args.compound_column,
            supplementary_column=args.supplementary_column,
        )
        print(f"Saved {len(result)} joined rows to {output}")
    elif args.command == "train":
        from training.train_pycaret import train_classifier

        output = train_classifier(
            args.input, args.output_dir, model_name=args.model,
            group_column=args.group_column, test_size=args.test_size,
            folds=args.folds, seed=args.seed, jobs=args.jobs,
            protocol=args.protocol, supplementary_column=args.supplementary_column,
            supplementary_boost=args.supplementary_boost,
            redundancy_offset=args.redundancy_offset, mixup_alpha=args.mixup_alpha,
            mixup_multiplier=args.mixup_multiplier,
        )
        print(f"Saved model, CV results, and holdout predictions to {output}")
    elif args.command == "audit":
        from training.audit_leakage import print_audit

        print_audit(args.input, args.run_dir)


if __name__ == "__main__":
    main()
