from pathlib import Path
import sys
import pandas as pd

from extract_phone_image_features import extract

ROOT = Path.cwd()
MANIFEST = ROOT / "analysis_output" / "phone_rgb_batch_manifest.csv"
OUTPUT_DIR = ROOT / "analysis_output" / "phone_rgb_features" / "B2"
RAW_ROOT = ROOT / "research_data" / "phone_photo" / "raw"

EXPECTED_COUNT = 999
EXPECTED_FEATURE_COUNT = 86
CONCENTRATIONS = ["M00", "M10", "M20", "M30", "M40"]

if not MANIFEST.exists():
    raise SystemExit(f"Manifest not found: {MANIFEST}")

manifest = pd.read_csv(MANIFEST)
required = {"batch", "concentration", "dilution_percent", "zoom", "file_path"}
missing = required - set(manifest.columns)

if missing:
    raise SystemExit(f"Missing manifest columns: {sorted(missing)}")

b2 = manifest[manifest["batch"] == "B2"].copy()

if len(b2) != EXPECTED_COUNT:
    raise SystemExit(
        f"Expected {EXPECTED_COUNT} B2 images, found {len(b2)}. "
        "Check the manifest before proceeding."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_rows = []
errors = []

for concentration in CONCENTRATIONS:
    subset = b2[b2["concentration"] == concentration]
    rows = []

    for _, item in subset.iterrows():
        path = Path(str(item["file_path"]))

        if not path.is_absolute():
            path = ROOT / path

        try:
            features = extract(path)

            if features is None:
                errors.append(f"Unreadable image: {path}")
                continue

            features["batch"] = "B2"
            features["concentration"] = concentration
            features["dilution_percent"] = int(item["dilution_percent"])
            features["zoom"] = item["zoom"]
            features["relative_path"] = str(path.relative_to(RAW_ROOT))

            rows.append(features)

        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if rows:
        df = pd.DataFrame(rows)

        feature_columns = [
            c for c in df.columns
            if c not in {
                "file_name", "relative_path", "batch",
                "concentration", "dilution_percent", "zoom"
            }
        ]

        if len(feature_columns) != EXPECTED_FEATURE_COUNT:
            raise SystemExit(
                f"{concentration}: expected 86 features, "
                f"found {len(feature_columns)}"
            )

        df.to_csv(OUTPUT_DIR / f"{concentration}.csv", index=False)
        all_rows.extend(rows)

        print(f"{concentration}: {len(rows)} processed")

    else:
        print(f"{concentration}: NO successful images")

if all_rows:
    combined = pd.DataFrame(all_rows)
    combined.to_csv(OUTPUT_DIR / "B2_all_features.csv", index=False)

print("\n===== B2 EXTRACTION REPORT =====")
print(f"Expected images: {EXPECTED_COUNT}")
print(f"Processed images: {len(all_rows)}")
print(f"Failed images: {len(errors)}")

if errors:
    error_file = OUTPUT_DIR / "extraction_errors.txt"
    error_file.write_text("\n".join(errors), encoding="utf-8")
    print(f"Errors saved: {error_file}")

if len(all_rows) != EXPECTED_COUNT or errors:
    print("\nWARNING: Extraction is incomplete. Do not start model training yet.")
    sys.exit(1)

print(f"Feature count: {EXPECTED_FEATURE_COUNT}")
print(f"Output folder: {OUTPUT_DIR}")
print("B2 extraction completed successfully.")
