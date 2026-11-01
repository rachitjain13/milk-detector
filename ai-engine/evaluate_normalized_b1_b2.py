
#!/usr/bin/env python
"""
MilkOptic AI
Normalized B1 training -> B2 external evaluation.

Uses the same ExtraTrees parameters as the original experiment.
B2 is evaluation-only; it is not used for model fitting or tuning.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "analysis_output/normalized_2448_features.csv"
)

MODEL_PATH = Path(
    "models/phone_rgb_b1_normalized_extratrees.joblib"
)

PREDICTIONS_PATH = Path(
    "analysis_output/normalized_b1_b2_predictions.csv"
)

METRICS_PATH = Path(
    "analysis_output/normalized_b1_b2_metrics.csv"
)

BASELINE_MAE = 11.2385
BASELINE_RMSE = 12.9506
BASELINE_R2 = 0.1622

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("MILKOPTIC AI - NORMALIZED B1 TO B2 EVALUATION")
print("=" * 65)

print("Total rows:", len(df))


# ============================================================
# SELECT IMAGE FEATURES
# ============================================================

excluded_columns = [
    "file_name",
    "image_width_original",
    "image_height_original",
    "batch",
    "concentration",
    "dilution_percent",
    "zoom",
    "file_path"
]

feature_columns = [
    col
    for col in df.select_dtypes(include="number").columns
    if col not in excluded_columns
]

print("\nNumber of image features:", len(feature_columns))

if not feature_columns:
    raise ValueError("No image features found.")

if df[feature_columns].isna().sum().sum() > 0:
    raise ValueError("Feature dataset contains missing values.")

if not np.isfinite(df[feature_columns].to_numpy()).all():
    raise ValueError("Feature dataset contains non-finite values.")


# ============================================================
# SPLIT B1 AND B2
# ============================================================

b1 = df[df["batch"] == "B1"].copy()
b2 = df[df["batch"] == "B2"].copy()

if b1.empty or b2.empty:
    raise ValueError("B1 or B2 dataset is empty.")

X_train = b1[feature_columns]
y_train = b1["dilution_percent"]

X_test = b2[feature_columns]
y_test = b2["dilution_percent"]

print("\nTraining images (B1):", len(b1))
print("Evaluation images (B2):", len(b2))


# ============================================================
# TRAIN MODEL
# ============================================================

model = ExtraTreesRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nTraining ExtraTrees model...")

model.fit(X_train, y_train)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "training_batch": "B1",
        "evaluation_batch": "B2",
        "image_size": "2448x2448"
    },
    MODEL_PATH
)

print("Model saved:", MODEL_PATH)


# ============================================================
# EVALUATE ON B2
# ============================================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n" + "=" * 65)
print("OVERALL B2 RESULTS")
print("=" * 65)

print(f"MAE :  {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R2  :  {r2:.4f}")


# ============================================================
# COMPARE WITH ORIGINAL BASELINE
# ============================================================

print("\n" + "=" * 65)
print("ORIGINAL VS NORMALIZED")
print("=" * 65)

print(f"{'Metric':<12}{'Original':>15}{'Normalized':>15}")

print(
    f"{'MAE':<12}"
    f"{BASELINE_MAE:>15.4f}"
    f"{mae:>15.4f}"
)

print(
    f"{'RMSE':<12}"
    f"{BASELINE_RMSE:>15.4f}"
    f"{rmse:>15.4f}"
)

print(
    f"{'R2':<12}"
    f"{BASELINE_R2:>15.4f}"
    f"{r2:>15.4f}"
)

print("\nMAE change:", round(mae - BASELINE_MAE, 4))
print("RMSE change:", round(rmse - BASELINE_RMSE, 4))
print("R2 change:", round(r2 - BASELINE_R2, 4))


# ============================================================
# RESULTS BY CONCENTRATION
# ============================================================

results = b2[
    ["concentration", "dilution_percent", "zoom", "file_path"]
].copy()

results["predicted_percent"] = predictions
results["absolute_error"] = np.abs(
    results["dilution_percent"] - predictions
)

print("\n" + "=" * 65)
print("RESULTS BY CONCENTRATION")
print("=" * 65)

concentration_results = results.groupby(
    ["concentration", "dilution_percent"]
).agg(
    image_count=("predicted_percent", "size"),
    actual_mean=("dilution_percent", "mean"),
    predicted_mean=("predicted_percent", "mean"),
    MAE=("absolute_error", "mean"),
    RMSE=(
        "absolute_error",
        lambda x: np.sqrt(np.mean(x ** 2))
    )
).reset_index()

print(
    concentration_results.round(4).to_string(index=False)
)


# ============================================================
# RESULTS BY ZOOM
# ============================================================

print("\n" + "=" * 65)
print("RESULTS BY ZOOM")
print("=" * 65)

zoom_results = results.groupby("zoom").agg(
    image_count=("predicted_percent", "size"),
    MAE=("absolute_error", "mean"),
    RMSE=(
        "absolute_error",
        lambda x: np.sqrt(np.mean(x ** 2))
    )
)

print(zoom_results.round(4).to_string())


# ============================================================
# SAVE RESULTS
# ============================================================

PREDICTIONS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(
    PREDICTIONS_PATH,
    index=False
)

metrics = pd.DataFrame([
    {
        "dataset": "Original",
        "MAE": BASELINE_MAE,
        "RMSE": BASELINE_RMSE,
        "R2": BASELINE_R2
    },
    {
        "dataset": "Normalized_2448",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
])

metrics.to_csv(
    METRICS_PATH,
    index=False
)

print("\nSaved predictions:", PREDICTIONS_PATH)
print("Saved metrics:", METRICS_PATH)

print("\nEVALUATION COMPLETED")
print("=" * 65)
