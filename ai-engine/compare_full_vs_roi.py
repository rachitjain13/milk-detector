
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FULL_CSV = (
    BASE_DIR
    / "analysis_output"
    / "camera_experiments"
    / "unified_all_batches.csv"
)

ROI_CSV = (
    BASE_DIR
    / "analysis_output"
    / "milk_roi_experiment"
    / "milk_roi_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "analysis_output"
    / "milk_roi_experiment"
    / "model_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1
}

TARGET = "dilution_percent"

GROUP = "concentration"

JOIN_KEYS = [
    "batch",
    "concentration",
    "relative_path"
]

EXCLUDE_FEATURES = [
    "dilution_percent",
    "image_width_original",
    "image_height_original",
    "batch",
    "concentration",
    "file_name",
    "relative_path",
    "camera_mode"
]


# ============================================================
# LOAD AND ALIGN DATA
# ============================================================

def load_data():

    print("\nLoading datasets...")

    full = pd.read_csv(FULL_CSV)
    roi = pd.read_csv(ROI_CSV)

    # Keep B1 only
    full = full[full["batch"] == "B1"].copy()
    roi = roi[roi["batch"] == "B1"].copy()

    # Check duplicates
    if full.duplicated(JOIN_KEYS).any():
        raise ValueError("Duplicate keys found in full-image data.")

    if roi.duplicated(JOIN_KEYS).any():
        raise ValueError("Duplicate keys found in ROI data.")

    # Confirm both datasets contain the same images
    aligned = full[JOIN_KEYS].merge(
        roi[JOIN_KEYS],
        on=JOIN_KEYS,
        how="outer",
        indicator=True
    )

    unmatched = aligned[aligned["_merge"] != "both"]

    if len(unmatched) > 0:
        raise ValueError(
            f"Dataset alignment failed. Unmatched rows: {len(unmatched)}"
        )

    # Keep matching row order
    full = full.sort_values(JOIN_KEYS).reset_index(drop=True)
    roi = roi.sort_values(JOIN_KEYS).reset_index(drop=True)

    print(f"Full-image B1 rows: {len(full)}")
    print(f"ROI B1 rows:        {len(roi)}")
    print(f"Matched images:     {len(aligned)}")

    if len(full) != len(roi):
        raise ValueError("Full and ROI image counts do not match.")

    return full, roi


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_features(df):

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    feature_columns = [
        col for col in numeric_columns
        if col not in EXCLUDE_FEATURES
    ]

    if not feature_columns:
        raise ValueError("No numeric features found.")

    X = df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X, feature_columns


# ============================================================
# CROSS VALIDATION
# ============================================================

def evaluate_model(df, X, feature_name):

    y = df[TARGET].to_numpy()
    groups = df[GROUP].to_numpy()

    logo = LeaveOneGroupOut()

    predictions = np.full(len(df), np.nan)

    print(f"\nEvaluating {feature_name}...")

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(**MODEL_PARAMS))
        ])

        model.fit(X_train, y_train)

        fold_predictions = model.predict(X_test)

        predictions[test_idx] = fold_predictions

        held_out = sorted(set(groups[test_idx]))

        print(
            f"Fold {fold}: held out {held_out}, "
            f"images={len(test_idx)}"
        )

    if np.isnan(predictions).any():
        raise ValueError("Some images did not receive OOF predictions.")

    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    r2 = r2_score(y, predictions)

    print(f"\n{feature_name} results")
    print(f"Features: {X.shape[1]}")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"R2:       {r2:.4f}")

    prediction_df = df[
        ["batch", "concentration", "dilution_percent",
         "zoom", "relative_path"]
    ].copy()

    prediction_df["predicted_percent"] = predictions

    prediction_df["absolute_error"] = np.abs(
        y - predictions
    )

    prediction_df["experiment"] = feature_name

    return {
        "experiment": feature_name,
        "images": len(df),
        "features": X.shape[1],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }, prediction_df


# ============================================================
# CONCENTRATION-WISE RESULTS
# ============================================================

def concentration_analysis(predictions):

    results = []

    for experiment in predictions["experiment"].unique():

        subset = predictions[
            predictions["experiment"] == experiment
        ]

        for concentration, group in subset.groupby("concentration"):

            results.append({
                "experiment": experiment,
                "concentration": concentration,
                "images": len(group),
                "MAE": group["absolute_error"].mean()
            })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" MILKOPTIC AI - FULL IMAGE VS MILK ROI")
    print("==========================================")

    full, roi = load_data()

    # Extract features separately
    X_full, full_features = get_features(full)
    X_roi, roi_features = get_features(roi)

    print("\nFeature counts:")
    print(f"Full-image features: {len(full_features)}")
    print(f"Milk-ROI features:   {len(roi_features)}")

    # Evaluate both approaches
    full_result, full_predictions = evaluate_model(
        full,
        X_full,
        "FULL_IMAGE"
    )

    roi_result, roi_predictions = evaluate_model(
        roi,
        X_roi,
        "MILK_ROI"
    )

    # Combined summary
    summary = pd.DataFrame([
        full_result,
        roi_result
    ])

    print("\n==========================================")
    print(" FINAL COMPARISON")
    print("==========================================")

    print(summary.to_string(index=False))

    # Save predictions
    all_predictions = pd.concat(
        [full_predictions, roi_predictions],
        ignore_index=True
    )

    all_predictions.to_csv(
        OUTPUT_DIR / "full_vs_roi_oof_predictions.csv",
        index=False
    )

    # Save summary
    summary.to_csv(
        OUTPUT_DIR / "full_vs_roi_summary.csv",
        index=False
    )

    # Save concentration-wise analysis
    concentration_results = concentration_analysis(
        all_predictions
    )

    concentration_results.to_csv(
        OUTPUT_DIR / "full_vs_roi_concentration_results.csv",
        index=False
    )

    print("\nSaved files:")
    print(OUTPUT_DIR / "full_vs_roi_summary.csv")
    print(OUTPUT_DIR / "full_vs_roi_oof_predictions.csv")
    print(OUTPUT_DIR / "full_vs_roi_concentration_results.csv")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
