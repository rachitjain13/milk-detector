from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path.cwd()
INPUT = ROOT / "analysis_output" / "phone_rgb_batches_combined.csv"
OUT = ROOT / "analysis_output"

if not INPUT.exists():
    raise SystemExit(f"Dataset not found: {INPUT}")

data = pd.read_csv(INPUT)

# Strictly use B1 only
data = data[data["batch"] == "B1"].copy().reset_index(drop=True)

# Physical sample group: both zoom levels belong to the same sample.
# This assumes each concentration folder represents one physical sample.
data["sample_group"] = data["concentration"].astype(str)

EXCLUDE = {
    "batch", "concentration", "dilution_percent",
    "zoom", "file_name", "relative_path", "sample_group"
}

features = [
    c for c in data.select_dtypes(include="number").columns
    if c not in EXCLUDE
]

if len(features) != 86:
    raise SystemExit(f"Expected 86 features, found {len(features)}")

if data[features].isna().any().any():
    raise SystemExit("Missing feature values detected.")

X = data[features]
y = data["dilution_percent"].astype(float)
groups = data["sample_group"]

logo = LeaveOneGroupOut()

predictions = {
    "All_86_features": np.zeros(len(data)),
    "Selected_10_features": np.zeros(len(data))
}

fold_results = []
selection_results = []

print("B1 images:", len(data))
print("Independent sample groups:", groups.nunique())
print("Features:", len(features))
print("\nStarting Leave-One-Group-Out validation...\n")

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups), start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    held_out = groups.iloc[test_idx].iloc[0]

    # Model 1: all 86 features
    model_all = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model_all.fit(X_train, y_train)
    pred_all = model_all.predict(X_test)
    predictions["All_86_features"][test_idx] = pred_all

    # Model 2: select top 10 features using training fold only
    selector_model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    selector_model.fit(X_train, y_train)

    selector = SelectFromModel(
        selector_model,
        threshold=-np.inf,
        max_features=10,
        prefit=True
    )

    selected_features = X.columns[selector.get_support()].tolist()

    model_selected = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model_selected.fit(X_train[selected_features], y_train)
    pred_selected = model_selected.predict(
        X_test[selected_features]
    )

    predictions["Selected_10_features"][test_idx] = pred_selected

    selection_results.append({
        "held_out_sample": held_out,
        "selected_features": ", ".join(selected_features)
    })

    for model_name, pred in [
        ("All_86_features", pred_all),
        ("Selected_10_features", pred_selected)
    ]:
        fold_results.append({
            "fold": fold,
            "held_out_sample": held_out,
            "model": model_name,
            "images_tested": len(test_idx),
            "actual_dilution": float(y_test.iloc[0]),
            "mean_predicted_dilution": float(np.mean(pred)),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred))
        })

    print(f"Fold {fold}: held out {held_out}")
    print("  Training images:", len(train_idx))
    print("  Test images:", len(test_idx))
    print("  Selected features:", selected_features)

# Final pooled predictions
prediction_frames = []

for model_name, pred in predictions.items():
    prediction_frames.append(pd.DataFrame({
        "batch": "B1",
        "sample_group": groups,
        "actual": y,
        "predicted": pred,
        "model": model_name,
        "zoom": data["zoom"].values
    }))

all_predictions = pd.concat(prediction_frames, ignore_index=True)

# Overall metrics
summary = []

for model_name, pred in predictions.items():
    summary.append({
        "model": model_name,
        "images": len(y),
        "independent_groups": groups.nunique(),
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred)
    })

summary_df = pd.DataFrame(summary)

# Save outputs
summary_df.to_csv(
    OUT / "b1_logo_model_summary.csv", index=False
)

pd.DataFrame(fold_results).to_csv(
    OUT / "b1_logo_fold_results.csv", index=False
)

pd.DataFrame(selection_results).to_csv(
    OUT / "b1_logo_selected_features.csv", index=False
)

all_predictions.to_csv(
    OUT / "b1_logo_predictions.csv", index=False
)

print("\n===== B1 LOGO VALIDATION SUMMARY =====")
print(summary_df.to_string(index=False))

print("\n===== PER-SAMPLE RESULTS =====")
print(pd.DataFrame(fold_results).to_string(index=False))

print("\nFiles saved:")
print("analysis_output/b1_logo_model_summary.csv")
print("analysis_output/b1_logo_fold_results.csv")
print("analysis_output/b1_logo_selected_features.csv")
print("analysis_output/b1_logo_predictions.csv")

print("\nB1 validation completed. B2 was not used.")
