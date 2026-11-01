
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "analysis_output" / "camera_experiments"
OUTPUT_DIR = DATA_DIR / "oof_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "12MP": DATA_DIR / "experiment_12MP_B1.csv",
    "COMBINED": DATA_DIR / "experiment_combined_B1.csv",
}

warnings.filterwarnings("ignore", category=UserWarning)


def run_experiment(name, file_path):
    print(f"\n{'=' * 55}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 55}")

    df = pd.read_csv(file_path)

    required = ["dilution_percent", "concentration", "zoom"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    excluded = {
        "dilution_percent",
        "image_width_original",
        "image_height_original",
    }

    metadata = {
        "batch",
        "concentration",
        "zoom",
        "file_name",
        "relative_path",
        "camera_mode",
    }

    feature_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in excluded
    ]

    if not feature_cols:
        raise ValueError("No usable numeric features found.")

    X = df[feature_cols]
    y = df["dilution_percent"].astype(float)
    groups = df["concentration"]

    logo = LeaveOneGroupOut()
    predictions = []

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups), start=1
    ):
        # Imputation is fitted only on the training fold.
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        )

        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted = model.predict(X.iloc[test_idx])

        fold_result = df.iloc[test_idx].copy()
        fold_result["predicted_percent"] = predicted
        fold_result["absolute_error"] = np.abs(
            fold_result["dilution_percent"] - predicted
        )
        fold_result["fold"] = fold
        predictions.append(fold_result)

        fold_mae = mean_absolute_error(
            y.iloc[test_idx], predicted
        )

        print(
            f"Fold {fold} | "
            f"Held out: {groups.iloc[test_idx].iloc[0]} | "
            f"MAE: {fold_mae:.3f}"
        )

    oof = pd.concat(predictions, ignore_index=True)

    output_file = OUTPUT_DIR / f"{name.lower()}_oof_predictions.csv"
    oof.to_csv(output_file, index=False)

    overall_mae = mean_absolute_error(
        oof["dilution_percent"], oof["predicted_percent"]
    )
    overall_rmse = np.sqrt(
        mean_squared_error(
            oof["dilution_percent"], oof["predicted_percent"]
        )
    )
    overall_r2 = r2_score(
        oof["dilution_percent"], oof["predicted_percent"]
    )

    print("\nOverall metrics:")
    print(f"Images: {len(oof)}")
    print(f"Features: {len(feature_cols)}")
    print(f"MAE:  {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"R2:   {overall_r2:.4f}")

    return oof


all_predictions = []

for experiment_name, path in EXPERIMENTS.items():
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    result = run_experiment(experiment_name, path)
    result["experiment"] = experiment_name
    all_predictions.append(result)


combined = pd.concat(all_predictions, ignore_index=True)

# Concentration-wise comparison
summary = (
    combined.groupby(["experiment", "concentration"])
    .agg(
        images=("absolute_error", "size"),
        actual_percent=("dilution_percent", "first"),
        mean_predicted=("predicted_percent", "mean"),
        MAE=("absolute_error", "mean"),
        RMSE=(
            "absolute_error",
            lambda x: np.sqrt(np.mean(np.square(x)))
        ),
    )
    .reset_index()
)

summary_file = OUTPUT_DIR / "concentration_wise_comparison.csv"
summary.to_csv(summary_file, index=False)

print("\n" + "=" * 55)
print("CONCENTRATION-WISE COMPARISON")
print("=" * 55)
print(summary.to_string(index=False))

print("\nSaved files:")
print(OUTPUT_DIR / "12mp_oof_predictions.csv")
print(OUTPUT_DIR / "combined_oof_predictions.csv")
print(summary_file)
