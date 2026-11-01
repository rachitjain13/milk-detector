from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path.cwd()
INPUT = ROOT / "analysis_output" / "phone_rgb_batches_combined.csv"
OUT = ROOT / "analysis_output"
MODEL_DIR = ROOT / "models"

data = pd.read_csv(INPUT)

b1 = data[data["batch"] == "B1"].copy()
b2 = data[data["batch"] == "B2"].copy()

metadata = {
    "batch", "concentration", "dilution_percent",
    "zoom", "file_name", "relative_path"
}

features = sorted(
    set(b1.select_dtypes(include="number").columns) - metadata
)

b2_features = sorted(
    set(b2.select_dtypes(include="number").columns) - metadata
)

if features != b2_features:
    raise SystemExit("B1/B2 feature schema mismatch.")

if len(features) != 86:
    raise SystemExit(f"Expected 86 features, found {len(features)}")

if len(b1) != 1002 or len(b2) != 999:
    raise SystemExit("Unexpected B1/B2 image count.")

X_train = b1[features]
y_train = b1["dilution_percent"].astype(float)

X_test = b2[features]
y_test = b2["dilution_percent"].astype(float)

if X_train.isna().any().any() or X_test.isna().any().any():
    raise SystemExit("Missing feature values detected.")

# Train using B1 only. B2 is not used for model selection or tuning.
model = ExtraTreesRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("Training on B1...")
print("Training images:", len(b1))
print("Training features:", len(features))

model.fit(X_train, y_train)

# First and only evaluation on B2
print("\nEvaluating frozen model on B2...")
predictions = model.predict(X_test)

overall = {
    "train_batch": "B1",
    "test_batch": "B2",
    "test_images": len(b2),
    "features": len(features),
    "MAE": mean_absolute_error(y_test, predictions),
    "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
    "R2": r2_score(y_test, predictions)
}

print("\n===== B2 OVERALL RESULTS =====")
for key, value in overall.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

results = b2[
    ["batch", "concentration", "dilution_percent",
     "zoom", "file_name", "relative_path"]
].copy()

results["actual"] = y_test.values
results["predicted"] = predictions
results["absolute_error"] = np.abs(y_test.values - predictions)

# Metrics by concentration
concentration_results = []

for concentration, group in results.groupby("concentration"):
    concentration_results.append({
        "concentration": concentration,
        "images": len(group),
        "actual_dilution": group["actual"].iloc[0],
        "mean_predicted": group["predicted"].mean(),
        "MAE": mean_absolute_error(group["actual"], group["predicted"]),
        "RMSE": np.sqrt(mean_squared_error(
            group["actual"], group["predicted"]
        ))
    })

concentration_df = pd.DataFrame(concentration_results)

# Metrics by zoom
zoom_results = []

for zoom, group in results.groupby("zoom"):
    zoom_results.append({
        "zoom": zoom,
        "images": len(group),
        "MAE": mean_absolute_error(group["actual"], group["predicted"]),
        "RMSE": np.sqrt(mean_squared_error(
            group["actual"], group["predicted"]
        ))
    })

zoom_df = pd.DataFrame(zoom_results)

OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": model,
    "features": features,
    "target": "dilution_percent",
    "training_batch": "B1",
    "model_type": "ExtraTreesRegressor",
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "random_state": 42
}, MODEL_DIR / "phone_rgb_b1_extratrees.joblib")

pd.DataFrame([overall]).to_csv(
    OUT / "b1_to_b2_overall_metrics.csv", index=False
)

concentration_df.to_csv(
    OUT / "b1_to_b2_concentration_metrics.csv", index=False
)

zoom_df.to_csv(
    OUT / "b1_to_b2_zoom_metrics.csv", index=False
)

results.to_csv(
    OUT / "b1_to_b2_predictions.csv", index=False
)

print("\n===== B2 CONCENTRATION RESULTS =====")
print(concentration_df.to_string(index=False))

print("\n===== B2 ZOOM RESULTS =====")
print(zoom_df.to_string(index=False))

print("\nSaved model:")
print("models/phone_rgb_b1_extratrees.joblib")

print("\nSaved evaluation files:")
print("analysis_output/b1_to_b2_overall_metrics.csv")
print("analysis_output/b1_to_b2_concentration_metrics.csv")
print("analysis_output/b1_to_b2_zoom_metrics.csv")
print("analysis_output/b1_to_b2_predictions.csv")

print("\nB1-to-B2 evaluation completed.")
print("Do not tune the model using these B2 results.")
