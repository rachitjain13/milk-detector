from pathlib import Path
import sys
import pandas as pd

ROOT = Path.cwd()
OUT_DIR = ROOT / "analysis_output"
B2_DIR = OUT_DIR / "phone_rgb_features" / "B2"

CONCENTRATIONS = {
    "M00": 0,
    "M10": 10,
    "M20": 20,
    "M30": 30,
    "M40": 40,
}

EXPECTED_B1 = 1002
EXPECTED_B2 = 999
EXPECTED_FEATURES = 86

b1_frames = []

# Load original B1 feature files
for concentration, dilution in CONCENTRATIONS.items():
    path = OUT_DIR / f"{concentration}_phone_features.csv"

    if not path.exists():
        raise SystemExit(f"B1 feature file missing: {path}")

    df = pd.read_csv(path)

    df["batch"] = "B1"
    df["concentration"] = concentration
    df["dilution_percent"] = dilution

    # Existing B1 relative_path starts with zoom folder.
    df["zoom"] = df["relative_path"].astype(str).str.replace(
        "\\", "/", regex=False
    ).str.split("/").str[0]

    b1_frames.append(df)

b1 = pd.concat(b1_frames, ignore_index=True)

# Load B2 features
b2_path = B2_DIR / "B2_all_features.csv"

if not b2_path.exists():
    raise SystemExit(f"B2 combined feature file missing: {b2_path}")

b2 = pd.read_csv(b2_path)

# Determine feature columns, excluding metadata
metadata = {
    "file_name", "relative_path", "batch",
    "concentration", "dilution_percent", "zoom"
}

b1_features = set(b1.columns) - metadata
b2_features = set(b2.columns) - metadata

print("B1 images:", len(b1))
print("B2 images:", len(b2))
print("B1 numeric feature columns:", len(b1_features))
print("B2 numeric feature columns:", len(b2_features))

if b1_features != b2_features:
    print("\nFeature mismatch detected.")
    print("Only in B1:", sorted(b1_features - b2_features))
    print("Only in B2:", sorted(b2_features - b1_features))
    sys.exit(1)

if len(b1_features) != EXPECTED_FEATURES:
    raise SystemExit("Expected exactly 86 common features.")

if len(b1) != EXPECTED_B1:
    raise SystemExit(f"Expected 1002 B1 rows, found {len(b1)}")

if len(b2) != EXPECTED_B2:
    raise SystemExit(f"Expected 999 B2 rows, found {len(b2)}")

combined = pd.concat([b1, b2], ignore_index=True)

# Convert feature columns to numeric
feature_columns = sorted(b1_features)

for col in feature_columns:
    combined[col] = pd.to_numeric(combined[col], errors="coerce")

if combined[feature_columns].isna().any().any():
    bad = combined[feature_columns].isna().sum()
    print("\nMissing/non-numeric feature values:")
    print(bad[bad > 0])
    raise SystemExit("Fix missing feature values before training.")

# Validate metadata
print("\n===== BATCH COUNTS =====")
print(combined.groupby("batch").size())

print("\n===== CONCENTRATION COUNTS =====")
print(combined.groupby(["batch", "concentration"]).size())

print("\n===== ZOOM COUNTS =====")
print(combined.groupby(["batch", "zoom"]).size())

# Check zoom values
if not set(combined["zoom"].astype(str)).issubset({"0.5", "1.0"}):
    raise SystemExit("Unexpected zoom values detected.")

# Save
output = OUT_DIR / "phone_rgb_batches_combined.csv"

# Put metadata first, followed by 86 features
columns = [
    "batch", "concentration", "dilution_percent",
    "zoom", "file_name", "relative_path"
] + feature_columns

combined[columns].to_csv(output, index=False)

print("\n===== FINAL VALIDATION =====")
print("Total images:", len(combined))
print("Total numeric features:", len(feature_columns))
print("Missing feature values:", int(combined[feature_columns].isna().sum().sum()))
print("Saved:", output)
print("\nB1 + B2 combination completed successfully.")
