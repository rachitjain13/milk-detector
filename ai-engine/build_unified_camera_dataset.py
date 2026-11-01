
import pandas as pd
import numpy as np
import os

FEATURES_PATH = "analysis_output/phone_rgb_batches_combined.csv"
MANIFEST_PATH = "analysis_output/phone_rgb_batch_manifest.csv"
OUTPUT_DIR = "analysis_output/camera_experiments"

os.makedirs(OUTPUT_DIR, exist_ok=True)

features = pd.read_csv(FEATURES_PATH)
manifest = pd.read_csv(MANIFEST_PATH)

# Normalize matching fields
def normalize_path(value):
    return str(value).replace("\\", "/").split("/")[-1].lower()

def normalize_zoom(value):
    return f"{float(value):.1f}"

features["file_name"] = features["relative_path"].apply(normalize_path)
manifest["file_name"] = manifest["file_path"].apply(normalize_path)

for df in [features, manifest]:
    df["batch"] = df["batch"].astype(str).str.upper()
    df["concentration"] = df["concentration"].astype(str).str.upper()
    df["zoom_key"] = df["zoom"].apply(normalize_zoom)

keys = ["batch", "concentration", "zoom_key", "file_name"]

print("\n===== SOURCE AUDIT =====")
print("Feature rows:", len(features))
print("Manifest rows:", len(manifest))

print("\nFeature duplicate keys:", features.duplicated(keys).sum())
print("Manifest duplicate keys:", manifest.duplicated(keys).sum())

print("\nManifest camera mode counts:")
print(manifest["camera_mode"].value_counts(dropna=False))

# Stop if matching is ambiguous
if features.duplicated(keys).any() or manifest.duplicated(keys).any():
    raise SystemExit("Duplicate matching keys found. Resolve before merging.")

# Attach camera metadata
unified = features.merge(
    manifest[keys + ["camera_mode", "format", "file_path"]],
    on=keys,
    how="left",
    validate="one_to_one",
    indicator=True
)

print("\n===== MERGE AUDIT =====")
print(unified["_merge"].value_counts())

if len(unified) != len(features):
    raise SystemExit("Unexpected row-count change during merge.")

if (unified["_merge"] != "both").any():
    print("\nUnmatched feature examples:")
    print(
        unified.loc[
            unified["_merge"] != "both",
            keys
        ].head(10).to_string(index=False)
    )
    raise SystemExit("Some feature rows did not match the manifest.")

unified.drop(columns=["_merge", "zoom_key"], inplace=True)

# Save complete unified dataset, preserving B1 and B2
unified.to_csv(
    f"{OUTPUT_DIR}/unified_all_batches.csv",
    index=False
)

# Keep B1 for model development
b1 = unified[unified["batch"] == "B1"].copy()

# Create the three experimental datasets
datasets = {
    "experiment_12MP_B1.csv": b1[b1["camera_mode"] == "12MP"],
    "experiment_108MP_B1.csv": b1[b1["camera_mode"] == "108MP"],
    "experiment_combined_B1.csv": b1
}

for filename, data in datasets.items():
    data.to_csv(
        f"{OUTPUT_DIR}/{filename}",
        index=False
    )

# Keep B2 separate
b2 = unified[unified["batch"] == "B2"].copy()

b2.to_csv(
    f"{OUTPUT_DIR}/B2_separate_exploratory.csv",
    index=False
)

print("\n===== EXPERIMENT DATASET SUMMARY =====")

for name, data in datasets.items():
    print(f"\n{name}")
    print("Images:", len(data))
    print("Camera modes:")
    print(data["camera_mode"].value_counts().to_dict())
    print("Concentrations:")
    print(data["concentration"].value_counts().to_dict())

print("\nB2 separate images:", len(b2))

print("\nAll datasets created successfully.")
print("Output folder:", OUTPUT_DIR)
