"""
MilkOptic AI - V7 Cross-Batch ROI Validation

Purpose:
- Train only on B1.
- Evaluate on B2 for the full-image baseline and V6 central ellipse ROI features.
- Reuse V6 extracted features; do not reprocess images.
- Keep all V6 outputs unchanged.

IMPORTANT:
B2 has already been used in earlier exploratory evaluation. Treat this as
exploratory cross-batch validation, NOT an untouched independent test set.
The central ellipse is a fixed geometric ROI, not a true milk segmentation mask.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent

INPUT_CSV = (
    ROOT / "analysis_output" / "camera_experiments" / "unified_all_batches.csv"
)
ROI_CSV = (
    ROOT / "analysis_output" / "milk_segmentation_v6_roi" / "v6_roi_features.csv"
)
OUT_ROOT = ROOT / "analysis_output" / "milk_segmentation_v7_cross_batch"

ROI_SIZES = (60,  70, 80)
RANDOM_STATE = 42

# These columns identify an image; they must never be model inputs.
IMAGE_KEYS = [
    "batch",
    "concentration",
    "dilution_percent",
    "zoom",
    "file_name",
    "relative_path",
]

FULL_METADATA = set(IMAGE_KEYS) | {
    "format",
    "camera_mode",
    "image_width",
    "image_height",
    "width",
    "height",
    "row_index",
}

ROI_METADATA = set(IMAGE_KEYS) | {"roi_size"}


def normalize_keys(data, label, unique_columns=None):
    """Normalize image keys consistently across CSV numeric dtypes."""
    missing = [c for c in IMAGE_KEYS if c not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing key columns: {missing}")

    result = data.copy()
    for column in IMAGE_KEYS:
        if column in ("dilution_percent", "zoom"):
            numeric = pd.to_numeric(result[column], errors="coerce")
            if numeric.isna().any():
                raise ValueError(f"{label} has invalid numeric key values in {column}.")
            result[column] = numeric.map(lambda value: format(float(value), ".12g"))
        else:
            result[column] = result[column].astype("string").str.strip()

    if result[IMAGE_KEYS].isna().any().any():
        raise ValueError(f"{label} contains missing image-key values.")

    check_columns = unique_columns if unique_columns is not None else IMAGE_KEYS
    if result.duplicated(check_columns).any():
        examples = result.loc[
            result.duplicated(check_columns, keep=False), check_columns
        ].head(10)
        raise ValueError(
            f"{label} has duplicate image keys. Examples:\\n{examples}"
        )
    return result


def get_feature_columns(data, metadata_columns):
    columns = []
    for column in data.columns:
        if column in metadata_columns:
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            columns.append(column)
    if not columns:
        raise ValueError("No numeric feature columns were found.")
    return columns


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def calculate_metrics(experiment, y_true, y_pred, n_features):
    return {
        "experiment": experiment,
        "train_batch": "B1",
        "test_batch": "B2",
        "n_train": int(len(y_true) if False else 0),  # filled by caller
        "n_test": int(len(y_true)),
        "n_features": int(n_features),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "validation": "B1 train -> B2 test (exploratory; B2 previously evaluated)",
    }


def run_experiment(
    experiment_name,
    train_data,
    test_data,
    feature_columns,
    output_metadata,
):
    X_train = train_data[feature_columns].replace([np.inf, -np.inf], np.nan)
    X_test = test_data[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_train = train_data["dilution_percent"].astype(float).to_numpy()
    y_test = test_data["dilution_percent"].astype(float).to_numpy()

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = calculate_metrics(
        experiment_name, y_test, predictions, len(feature_columns)
    )
    metrics["n_train"] = int(len(train_data))

    prediction_data = test_data[output_metadata].copy()
    prediction_data["experiment"] = experiment_name
    prediction_data["prediction"] = predictions
    prediction_data["absolute_error"] = np.abs(y_test - predictions)
    prediction_data["signed_error"] = predictions - y_test

    return metrics, prediction_data


def assert_same_image_rows(base_data, roi_data, batch, roi_size):
    """Ensure each image appears exactly once in the selected ROI table."""
    base = base_data[base_data["batch"].str.upper().eq(batch)].copy()
    roi = roi_data[
        roi_data["batch"].str.upper().eq(batch)
        & (pd.to_numeric(roi_data["roi_size"], errors="coerce") == roi_size)
    ].copy()

    base_keys = set(map(tuple, base[IMAGE_KEYS].to_numpy()))
    roi_keys = set(map(tuple, roi[IMAGE_KEYS].to_numpy()))

    missing = base_keys - roi_keys
    extra = roi_keys - base_keys
    if missing or extra or len(base) != len(roi):
        raise ValueError(
            f"Image mismatch for {batch}, ROI {roi_size}%: "
            f"baseline rows={len(base)}, ROI rows={len(roi)}, "
            f"missing ROI keys={len(missing)}, extra ROI keys={len(extra)}."
        )

    roi = base[IMAGE_KEYS].merge(
        roi, on=IMAGE_KEYS, how="left", validate="one_to_one"
    )
    if len(roi) != len(base):
        raise ValueError("Unexpected row-count change during ROI alignment.")
    return base, roi


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Unified dataset not found: {INPUT_CSV}")
    if not ROI_CSV.exists():
        raise FileNotFoundError(f"V6 ROI features not found: {ROI_CSV}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading unified dataset and V6 ROI features...")
    base = pd.read_csv(INPUT_CSV)
    roi_all = pd.read_csv(ROI_CSV)

    required_base = set(IMAGE_KEYS)
    missing_base = required_base - set(base.columns)
    if missing_base:
        raise ValueError(f"Unified CSV missing required columns: {sorted(missing_base)}")

    required_roi = set(IMAGE_KEYS) | {"roi_size"}
    missing_roi = required_roi - set(roi_all.columns)
    if missing_roi:
        raise ValueError(f"V6 ROI CSV missing required columns: {sorted(missing_roi)}")

    base = normalize_keys(base, "Unified dataset")
    roi_all = normalize_keys(
        roi_all,
        "V6 ROI features",
        unique_columns=IMAGE_KEYS + ["roi_size"],
    )

    base["batch"] = base["batch"].str.upper()
    roi_all["batch"] = roi_all["batch"].str.upper()
    roi_all["roi_size"] = pd.to_numeric(roi_all["roi_size"], errors="raise")

    if base["dilution_percent"].isna().any():
        raise ValueError("Unified dataset has missing dilution_percent labels.")
    if roi_all["dilution_percent"].isna().any():
        raise ValueError("ROI features have missing dilution_percent labels.")

    b1 = base[base["batch"].eq("B1")].copy()
    b2 = base[base["batch"].eq("B2")].copy()
    if b1.empty or b2.empty:
        raise ValueError(f"Expected B1 and B2 rows; got B1={len(b1)}, B2={len(b2)}.")

    # Baseline numeric image features, excluding all metadata/labels.
    full_feature_columns = get_feature_columns(base, FULL_METADATA)
    output_metadata = [c for c in IMAGE_KEYS if c in b2.columns]

    print(f"Unified rows: {len(base)} | B1 train: {len(b1)} | B2 test: {len(b2)}")
    print(f"Full-image feature count: {len(full_feature_columns)}")

    results = []
    prediction_frames = []

    # Full-image baseline.
    metrics, predictions = run_experiment(
        "FULL_IMAGE_BASELINE",
        b1,
        b2,
        full_feature_columns,
        output_metadata,
    )
    results.append(metrics)
    prediction_frames.append(predictions)

    # V6 ROI models. Validate image-key alignment independently for both batches.
    for size in ROI_SIZES:
        b1_base, b1_roi = assert_same_image_rows(base, roi_all, "B1", size)
        b2_base, b2_roi = assert_same_image_rows(base, roi_all, "B2", size)

        roi_feature_columns = get_feature_columns(b1_roi, ROI_METADATA)

        # Ensure the test feature schema exactly matches the training schema.
        missing_test_features = set(roi_feature_columns) - set(b2_roi.columns)
        if missing_test_features:
            raise ValueError(
                f"ROI {size}% B2 missing features: {sorted(missing_test_features)}"
            )

        # Labels and metadata must agree after alignment.
        if not np.allclose(
            b1_base["dilution_percent"].astype(float).to_numpy(),
            b1_roi["dilution_percent"].astype(float).to_numpy(),
        ):
            raise ValueError(f"B1 labels differ between source and ROI data at {size}%.")
        if not np.allclose(
            b2_base["dilution_percent"].astype(float).to_numpy(),
            b2_roi["dilution_percent"].astype(float).to_numpy(),
        ):
            raise ValueError(f"B2 labels differ between source and ROI data at {size}%.")

        experiment_name = f"CENTRAL_ELLIPSE_{size}PCT"
        print(f"Running {experiment_name}...")
        metrics, predictions = run_experiment(
            experiment_name,
            b1_roi,
            b2_roi,
            roi_feature_columns,
            output_metadata,
        )
        results.append(metrics)
        prediction_frames.append(predictions)

    metrics_df = pd.DataFrame(results)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    concentration_errors = (
        predictions_df.groupby(
            ["experiment", "concentration", "dilution_percent"],
            dropna=False,
        )
        .agg(
            n=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            mean_prediction=("prediction", "mean"),
            prediction_std=("prediction", "std"),
            mean_signed_error=("signed_error", "mean"),
        )
        .reset_index()
    )

    metrics_path = OUT_ROOT / "v7_cross_batch_metrics.csv"
    predictions_path = OUT_ROOT / "v7_b2_predictions.csv"
    concentration_path = OUT_ROOT / "v7_concentration_wise_errors.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    concentration_errors.to_csv(concentration_path, index=False)

    print("\nV7 CROSS-BATCH RESULTS")
    print(metrics_df.to_string(index=False))
    print("\nCONCENTRATION-WISE B2 ERRORS")
    print(concentration_errors.to_string(index=False))

    print("\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {predictions_path}")
    print(f"  {concentration_path}")
    print(
        "\nIMPORTANT: B2 is an exploratory cross-batch evaluation, not an "
        "untouched independent test set. Fixed central ellipses are not "
        "true milk segmentation. Do not claim lab-grade adulteration detection."
    )


if __name__ == "__main__":
    main()
