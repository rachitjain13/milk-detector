
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = "analysis_output/camera_experiments"

EXPERIMENTS = {
    "12MP": f"{BASE}/experiment_12MP_B1.csv",
    "COMBINED": f"{BASE}/experiment_combined_B1.csv"
}

def evaluate(path, name):

    df = pd.read_csv(path)

    # Numeric feature columns
    exclude = {
        "dilution_percent",
        "image_width_original",
        "image_height_original"
    }

    features = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in exclude
    ]

    X = df[features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0)

    y = df["dilution_percent"].to_numpy()

    # Same physical sample's zoom images stay together
    groups = df["concentration"].to_numpy()

    logo = LeaveOneGroupOut()

    predictions = np.zeros(len(df))

    print(f"\n===== {name} =====")
    print("Images:", len(df))
    print("Features:", len(features))
    print("Groups:", len(np.unique(groups)))

    for train_idx, test_idx in logo.split(X, y, groups):

        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X.iloc[train_idx], y[train_idx])

        predictions[test_idx] = model.predict(X.iloc[test_idx])

        held_group = groups[test_idx][0]

        print(
            f"Held out: {held_group} | "
            f"MAE: {mean_absolute_error(y[test_idx], predictions[test_idx]):.3f}"
        )

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    try:
        r2 = r2_score(y, predictions)
    except Exception:
        r2 = float("nan")

    print("\nOverall results:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    return {
        "experiment": name,
        "images": len(df),
        "features": len(features),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

results = []

for name, path in EXPERIMENTS.items():
    results.append(evaluate(path, name))

summary = pd.DataFrame(results)

summary.to_csv(
    f"{BASE}/experiment_comparison_results.csv",
    index=False
)

print("\n===== FINAL COMPARISON =====")
print(summary.to_string(index=False))

print("\nSaved experiment comparison results.")
