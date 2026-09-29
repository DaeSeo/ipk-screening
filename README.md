# SMILES and image feature pipeline

This project supports `Data/` either beside `Main/` or inside `Main/`.
The default folder is the one containing the raw CSV (if both contain source
CSVs, the folder inside `Main/` takes precedence). All commands below
run from the **project root** (the directory containing both folders). On a
case-sensitive filesystem rename the downloaded `main/` folder to `Main/`.

## Install

Use Python 3.11 in a clean environment, then:

```bash
python -m pip install -r requirements.txt
```

`openpyxl` is required to read the 2024 Excel file.

The first embedding command downloads pretrained model weights from Hugging
Face. MoLFormer loads model code from its publisher (`trust_remote_code=True`)
and selects its Transformers 4 compatibility revision. Review the model repo if
your environment has restrictions on executing downloaded code.

## 1. Preprocess the CSV

### Two yearly compound screening files

The 2024 Excel input should have `Compound ID`, `Barcode`, `Well`, `SMILES`,
`infected cells / number of cells (%)`, and `cellratio(%)` columns. The 2025
CSV uses `Compound_ID` for the compound identifier; the other five headings
are the same. Read, align, concatenate, standardize SMILES, and label both
thresholds with:

```bash
python Main/main.py preprocess-screening \
  --input-2024 Data/OMICRON_2024_cpd.xlsx \
  --input-2025 Data/OMICRON_2025_cpd.csv \
  --target hit70 \
  --output Data/processed.csv \
  --combined-output Data/compound_screening_results.csv
```

Replace the two input filenames with your actual filenames. Missing, blank,
and `--` SMILES are removed before assigning `row_id`. Well names such as
`A01` become `A1`; the original years, plate, well, and (if present) compound
ID remain in the output. `hit70=1` means **both** `infection_percentage > 70`
and `cell_percentage > 70`; `hit50=1` means **both** are `> 50`. Equality at
70 or 50 is not a hit. Missing or nonnumeric measurements result in missing
hit labels, rather than 0. `activity` copies `hit70` by default for downstream
training. Use `--target hit50` to train on `hit50` (prefer a different
`--output` name if retaining both analyses). `Data/invalid_smiles.csv` lists
rows whose nonempty SMILES RDKit cannot standardize.

For these new inputs, continue with:

```bash
python Main/main.py embed-molecular --input Data/processed.csv --model chemberta --batch-size 16
python Main/main.py assemble --input Data/processed.csv --molecular chemberta --compound-column compound_id
python Main/main.py train --input Data/features/chemberta.parquet --protocol baseline --model xgboost --jobs 1 --output-dir Data/models/hit70_baseline
```

The numeric assay columns, hit labels, and compound ID do not enter the
predictor matrix; only embeddings are used. If you change only `--target`,
reuse the molecular embeddings and rerun `assemble` and `train` with separately
named feature/model outputs. These labels use the **high infection** rule you
specified; confirm that this agrees with your biological definition of a hit.

### Older single CSV format

Put `combined_df.csv` in `Data/` with a `SMILES` column and an
`infected cells / number of cells (%)` column. Then run:

```bash
python Main/main.py preprocess --input Data/combined_df.csv --threshold 70 --positive-if below
```

If your prompt currently ends in `Main %` and the source is at
`Main/Data/combined_df.csv`, run from there:

```bash
python main.py preprocess --input Data/combined_df.csv --threshold 70 --positive-if below
```

Outputs and the following stage defaults will use that same `Main/Data/` folder.

`Data/processed.csv` includes the input columns, a stable `row_id`,
`standardized_smiles`, and binary `activity`. Invalid SMILES remain as blank
`standardized_smiles`; the CLI also saves those rows in
`Data/invalid_smiles.csv` for review. They are excluded from molecular embeddings. The default
label is **1 when infected-cell percentage is >= 70**. If `1` should mean
*low infection*, use `--positive-if below`. The threshold is inclusive in
both cases. Review whether the measured column is infection or inhibition
before interpreting model predictions.

For antiviral hit prediction from an *infected cell percentage*, use
`--positive-if below` so `activity=1` means low infection. The default
`--positive-if above` assigns `1` to high infection and is appropriate only
when that is the class you intend to predict. Decide the assay hit threshold
from the experimental protocol before training.

If there is exactly one source CSV in `Data/`, `python Main/main.py` still
runs preprocessing. The earlier `python Main/main.py --input ...` works too.

## 2. Molecular features

```bash
python Main/main.py embed-molecular --model chemberta --batch-size 16
python Main/main.py embed-molecular --model molformer --batch-size 8
```

Results: `Data/embeddings/molecular/chemberta.parquet` and
`molformer.parquet`. Both use attention-mask mean pooling over the final
hidden states (including special tokens), batch inference, model.eval(), and
float32 output. The default maximum token length is 200; longer SMILES are
truncated. Set `--pooling pooler` if the selected model exposes a pooler.

## 3. Image features (when images are ready)

Copy `Data/manifests/images.example.csv` to `Data/manifests/images.csv` and
replace the example entries. `row_id` must match `Data/processed.csv`;
`image_path` is relative to `Data/images/` (or absolute). Multiple image rows
may point to one `row_id`.

```bash
python Main/main.py embed-images --batch-size 8
```

The DINOv2 example converts each image to RGB, extracts its CLS token vector,
and averages multiple image vectors for a `row_id`. For multichannel
microscopy, first decide which channels and rendering convention are
biologically appropriate. Do not treat the RGB conversion as a validated
microscopy preprocessing pipeline.

## 4. Assemble and train

For three ChemBERTa/XGBoost experiments from the project root, run the
preprocessing and embedding stages once, then train each protocol separately:

```bash
python Main/main.py preprocess --input Data/combined_df.csv --threshold 70 --positive-if below
python Main/main.py embed-molecular --model chemberta --batch-size 16
python Main/main.py assemble --molecular chemberta

python Main/main.py train --input Data/features/chemberta.parquet --protocol baseline --model xgboost --seed 42 --jobs 1 --output-dir Data/models/chemberta_baseline
python Main/main.py train --input Data/features/chemberta.parquet --protocol scaffold --model xgboost --seed 42 --jobs 1 --output-dir Data/models/chemberta_scaffold
python Main/main.py train --input Data/features/chemberta.parquet --protocol manuscript --model xgboost --seed 42 --jobs 1 --output-dir Data/models/chemberta_manuscript
```

`baseline` uses a 20% SMILES-grouped holdout; `scaffold` and `manuscript`
use the same seeded, approximately 5% scaffold holdout and scaffold-grouped
cross-validation. `scaffold` trains plain PyCaret XGBoost without negative
filtering, weighting, or mixup. `manuscript` also filters redundant negatives,
weights positives, and creates mixup embeddings within each training fold.
The manuscript estimator uses a logistic XGBRegressor to support soft labels;
its hyperparameters differ from PyCaret's default XGBoost classifier, so the
scaffold/manuscript comparison does not isolate augmentation alone.
Check whether `--positive-if above` (default) matches the assay definition;
use `--positive-if below` during preprocessing if lower infection means hit.

Each training run prints holdout metrics and writes `holdout_hit_metrics.csv`:
F1, Precision, Recall, Accuracy, PR-AUC (sklearn Average Precision), ROC-AUC,
Balanced Accuracy, and confusion-matrix counts. These are computed for
`--hit-label 1` by default using `prediction_score_1` from PyCaret's
`raw_score=True` output, saved in `holdout_predictions.csv`. If using an
existing feature table where `activity=0` means hit, supply `--hit-label 0`
to measure hits correctly. The manuscript protocol's augmentation always
operates on class 1, so reprocess with `--positive-if below` before using it
for hit prediction. Embeddings do not need regeneration when only labels change;
rerun `assemble` and `train`.
The screenshot's `tau*` is a selected decision threshold, not an evaluation
metric. This implementation reports PyCaret's default classification threshold
(normally 0.5); matching `tau*` would require tuning the threshold on training
folds only, before evaluating the held-out test set.

To inspect a finished baseline run for sample overlap, label misalignment,
and an inflated F1 from an imbalanced positive class, run:

```bash
python Main/main.py audit --input Data/features/chemberta.parquet --run-dir Data/models/chemberta_baseline
```

This writes `leakage_audit.json` alongside the model. Zero overlaps and label
mismatches rule out direct row/SMILES reuse in that saved split; they cannot
rule out related compounds, batch effects, or data provenance problems.

Choose any available modalities:

```bash
python Main/main.py assemble --molecular chemberta
python Main/main.py train --input Data/features/chemberta.parquet --model rf
```

For both molecular models and images:

```bash
python Main/main.py assemble --molecular chemberta molformer --images
python Main/main.py train --input Data/features/chemberta_molformer_dinov2.parquet --model compare
```

`assemble` keeps rows with features from all requested modalities and uses
`row_id` for exact joins. `train` takes **only** columns beginning `cb_`,
`mf_`, or `img_` as predictors. IDs, raw SMILES, infection percentage, and
activity are excluded from predictors. PyCaret creates a group-disjoint
holdout and uses GroupKFold on the training portion. By default the group is
`standardized_smiles`; supply `--group-column YOUR_COMPOUND_ID` when an
assay compound identifier is more appropriate. For missing group values it
falls back to `row_id`, so inspect any missing compound identifiers.

Models `rf`, `lr`, and `xgboost` can be selected via `--model`. With
`--model compare`, PyCaret compares available `lr`, `rf`, `et`, and `xgboost`
models by cross-validated AUC. The default `--jobs 1` limits CPU
oversubscription on a Mac. The run writes a PyCaret model (`model.pkl`), CV
results, held-out predictions and metrics, and train/test row ID lists to
`Data/models/experiment/`. Do not use the holdout to tune settings repeatedly.

## ChemBERTa-only manuscript-inspired protocol

Start from a project root containing `Main/` and `Data/`. The four commands
below use **no images and no MolFormer**:

```bash
python Main/main.py preprocess --input Data/combined_df.csv --threshold 70 --positive-if above
python Main/main.py embed-molecular --model chemberta --batch-size 16
python Main/main.py assemble --molecular chemberta
python Main/main.py train --input Data/features/chemberta.parquet --protocol manuscript --jobs 1
```

Use `--positive-if below` if the original percentage measures *infected cells*
and a hit means *less infection*. If the source column is instead inhibition
or efficacy, `above` may be right. The direction must be checked against the
assay definition before reporting a hit rate.

The manuscript mode keeps the 5% test split and training CV scaffold-disjoint,
approximately stratified by hit status. Acyclic molecules have no Murcko
framework, so they are grouped by canonical SMILES. Within each training
fold, it calculates each non-hit molecule's nearest other non-hit cosine
similarity. Negatives with scores above mean minus 0.1 standard deviations
are removed. Remaining positive samples receive the negative/positive count
ratio as a training weight. Five mixed training examples per positive are
generated by default from a positive anchor and random training partner with
`lambda ~ Beta(0.4, 0.4)`. A custom XGBoost classifier wrapper uses a logistic
objective with soft mixup labels and is evaluated in PyCaret. The held-out
5% never participates in negative filtering, weights, or mixup.

The screenshot specifies the threshold *formula* but not the redundancy score,
mixup pair-selection rule, `alpha`, number of pairs, or supplementary-instance
weight. Thus this is a **defined implementation of the approach**, not an
exact reproduction of the counts in that screenshot. Control its assumptions
with `--redundancy-offset`, `--mixup-alpha`, `--mixup-multiplier`, `--folds`,
and `--test-size`. The exact nearest-negative calculation scales quadratically
and may take a long time on a laptop. Check `train_rows.csv`, `test_rows.csv`,
and `run.json` for class counts and settings. When PyCaret returns the fitted
custom estimator directly, `run.json` also records the final fit's filtered
non-hit count, mixup count, and positive weight.

To boost compounds with additional screening results, supply a **0/1** column
in your source CSV, for example `supplementary_screened`. Preserve it when
assembling and specify a multiplier when training:

```bash
python Main/main.py assemble --molecular chemberta --supplementary-column supplementary_screened
python Main/main.py train --input Data/features/chemberta.parquet --protocol manuscript \
  --supplementary-column supplementary_screened --supplementary-boost 2 --jobs 1
```

The `2` is only an example; choose the instance weighting rule from the actual
protocol and validate the presence and meaning of the flag in your CSV.

## Project layout

```text
Main/
  main.py
  processing/normalisation.py
  feature_extraction/common.py
  feature_extraction/assemble.py
  feature_extraction/molecular/{chemberta,molformer,extract}.py
  feature_extraction/image/dino.py
  training/train_pycaret.py
Data/
  combined_df.csv              # your input (not shipped)
  processed.csv                # generated
  images/                      # your images (not shipped)
  manifests/images.example.csv
  embeddings/{molecular,image}/
  features/
  models/
```
