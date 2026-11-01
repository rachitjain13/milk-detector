
#!/usr/bin/env python
"""
MilkOptic AI - Normalized Phone Image Feature Extraction

Purpose:
1. Read all normalized B1 and B2 images.
2. Extract the same 86 numerical image features.
3. Preserve batch, concentration, dilution and zoom metadata.
4. Save the combined normalized feature dataset.
5. Report extraction failures and data quality checks.

Original images and original feature datasets are not modified.
"""

from pathlib import Path
import pandas as pd

from extract_phone_image_features import extract


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_PATH = Path(
    "analysis_output/normalized_2448_manifest.csv"
)

OUTPUT_PATH = Path(
    "analysis_output/normalized_2448_features.csv"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = [
    "batch",
    "concentration",
    "dilution_percent",
    "zoom",
    "file_path"
]

missing_columns = [
    col for col in required_columns
    if col not in manifest.columns
]

if missing_columns:
    raise ValueError(
        f"Missing manifest columns: {missing_columns}"
    )

print("=" * 60)
print("MILKOPTIC AI - NORMALIZED FEATURE EXTRACTION")
print("=" * 60)

print("Manifest images:", len(manifest))


# ============================================================
# FEATURE EXTRACTION
# ============================================================

rows = []
failed = []

for i, row in manifest.iterrows():

    image_path = Path(row["file_path"])

    try:

        if not image_path.exists():
            raise FileNotFoundError("Image file does not exist")

        # Extract features using the existing extractor.
        # IMPORTANT: extract() expects a Path object.
        features = extract(image_path)

        if features is None:
            raise ValueError("OpenCV could not read image")

        # Add dataset metadata.
        features["batch"] = row["batch"]
        features["concentration"] = row["concentration"]
        features["dilution_percent"] = row["dilution_percent"]
        features["zoom"] = row["zoom"]
        features["file_path"] = str(image_path)

        rows.append(features)

    except Exception as error:

        failed.append({
            "file_path": str(image_path),
            "error": str(error)
        })

        print(
            f"\nERROR [{i + 1}/{len(manifest)}]: "
            f"{image_path}"
        )
        print("Reason:", error)

    if (i + 1) % 100 == 0:
        print(
            f"Processed {i + 1}/{len(manifest)} | "
            f"Successful: {len(rows)} | "
            f"Failed: {len(failed)}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

if not rows:
    raise RuntimeError(
        "No images were successfully processed."
    )

df = pd.DataFrame(rows)


# ============================================================
# SAVE FEATURE DATASET
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SAVE FAILURE REPORT
# ============================================================

if failed:

    failure_df = pd.DataFrame(failed)

    failure_df.to_csv(
        "analysis_output/normalized_2448_failures.csv",
        index=False
    )


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

metadata_columns = [
    "file_name",
    "image_width_original",
    "image_height_original",
    "batch",
    "concentration",
    "dilution_percent",
    "zoom",
    "file_path"
]

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

print("\n" + "=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)

print("Total manifest images:", len(manifest))
print("Successfully extracted:", len(df))
print("Failed images:", len(failed))
print("Total dataset columns:", len(df.columns))
print("Numeric columns:", len(numeric_columns))

print("\nBATCH DISTRIBUTION")

print(
    df.groupby(
        ["batch", "concentration"]
    ).size().to_string()
)

print("\nZOOM DISTRIBUTION")

print(
    df.groupby(
        ["batch", "zoom"]
    ).size().to_string()
)

print("\nIMAGE RESOLUTION CHECK")

print(
    df.groupby(
        ["batch", "image_width_original", "image_height_original"]
    ).size().to_string()
)

print("\nDATA QUALITY")

print("Missing values:", df.isna().sum().sum())

print(
    "Duplicate image paths:",
    df["file_path"].duplicated().sum()
)

print(
    "Non-finite numeric values:",
    (~df[numeric_columns].apply(
        lambda col: pd.Series(
            col.map(
                lambda value: (
                    pd.notna(value) and
                    abs(float(value)) != float("inf")
                )
            ).values,
            index=col.index
        )
    )).sum().sum()
)

print("\nSaved dataset:")
print(OUTPUT_PATH)

if failed:
    print(
        "\nFailure report:",
        "analysis_output/normalized_2448_failures.csv"
    )

print("\nFEATURE EXTRACTION COMPLETED")
print("=" * 60)
