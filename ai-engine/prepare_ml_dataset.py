
import pandas as pd
from pathlib import Path

samples = {
    "M00": 0,
    "M10": 10,
    "M20": 20,
    "M30": 30,
    "M40": 40,
}

frames = []

for sample, concentration in samples.items():
    path = Path(f"analysis_output/{sample}_phone_features.csv")
    df = pd.read_csv(path)

    # Keep only 1.0x camera images
    df["zoom"] = df["relative_path"].str.split("\\").str[0]
    df = df[df["zoom"] == "1.0"].copy()

    df["sample_id"] = sample
    df["water_added_pct"] = concentration

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# Remove image metadata and identifiers from model features
exclude = [
    "file_name",
    "relative_path",
    "zoom",
    "sample_id",
    "water_added_pct",
    "image_width_original",
    "image_height_original",
]

feature_columns = [
    col for col in data.select_dtypes(include="number").columns
    if col not in exclude
]

print("Dataset shape:", data.shape)
print("Feature count:", len(feature_columns))
print("\nSamples:")
print(data.groupby("sample_id").size())
print("\nMissing values:", data[feature_columns].isna().sum().sum())
print("Duplicate feature rows:", data[feature_columns].duplicated().sum())

output = Path("analysis_output/milkoptic_ml_dataset.csv")
data.to_csv(output, index=False)

print("\nSaved:", output)
