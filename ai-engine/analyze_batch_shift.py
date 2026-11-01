from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()
INPUT = ROOT / "analysis_output" / "phone_rgb_batches_combined.csv"
OUT = ROOT / "analysis_output"

df = pd.read_csv(INPUT)

metadata = {
    "batch", "concentration", "dilution_percent",
    "zoom", "file_name", "relative_path"
}

features = [
    c for c in df.select_dtypes(include="number").columns
    if c not in metadata
]

# Compare batches at matching concentration and zoom
rows = []

for concentration in sorted(df["concentration"].unique()):
    for zoom in sorted(df["zoom"].unique()):
        subset = df[
            (df["concentration"] == concentration) &
            (df["zoom"] == zoom)
        ]

        b1 = subset[subset["batch"] == "B1"]
        b2 = subset[subset["batch"] == "B2"]

        if len(b1) == 0 or len(b2) == 0:
            continue

        for feature in features:
            mean1 = b1[feature].mean()
            mean2 = b2[feature].mean()
            std1 = b1[feature].std()
            std2 = b2[feature].std()

            pooled = np.sqrt((std1**2 + std2**2) / 2)

            if pooled > 0:
                standardized_difference = (mean2 - mean1) / pooled
            else:
                standardized_difference = 0.0

            rows.append({
                "concentration": concentration,
                "zoom": zoom,
                "feature": feature,
                "B1_mean": mean1,
                "B2_mean": mean2,
                "B1_std": std1,
                "B2_std": std2,
                "standardized_difference": standardized_difference,
                "absolute_difference": abs(standardized_difference)
            })

result = pd.DataFrame(rows)

result.to_csv(
    OUT / "b1_b2_feature_distribution.csv",
    index=False
)

summary = (
    result.groupby("feature")["absolute_difference"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

summary.to_csv(
    OUT / "b1_b2_top_distribution_shifts.csv",
    index=False
)

print("Feature comparisons:", len(result))
print("\n===== TOP 20 DISTRIBUTION SHIFTS =====")
print(summary.head(20).to_string(index=False))

print("\nSaved:")
print("analysis_output/b1_b2_feature_distribution.csv")
print("analysis_output/b1_b2_top_distribution_shifts.csv")
