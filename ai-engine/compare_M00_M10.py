import pandas as pd
from pathlib import Path

files = {
    "M00": Path("analysis_output/M00_phone_features.csv"),
    "M10": Path("analysis_output/M10_phone_features.csv"),
}

frames = []

for sample, path in files.items():
    df = pd.read_csv(path)
    df["sample"] = sample
    df["zoom"] = df["relative_path"].str.split("\\").str[0]
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

features = [
    c for c in data.select_dtypes(include="number").columns
    if c not in ["image_width", "image_height"]
]

summary = data.groupby(["zoom", "sample"])[features].mean().round(3)

print("\n===== M00 vs M10 FEATURE COMPARISON =====")
for zoom in sorted(data["zoom"].dropna().unique()):
    print(f"\n--- {zoom} camera ---")
    print(summary.loc[zoom].T.to_string())

output = Path("analysis_output/M00_M10_comparison.csv")
summary.to_csv(output)
print(f"\nComparison saved: {output}")