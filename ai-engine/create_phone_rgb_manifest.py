
from pathlib import Path
import pandas as pd

ROOT = Path("research_data/phone_photo/raw")
OUTPUT = Path("analysis_output/phone_rgb_batch_manifest.csv")

DATASETS = {
    "B1": ["M00", "M10", "M20", "M30", "M40"],
    "B2": ["M00_B2", "M10_B2", "M20_B2", "M30_B2", "M40_B2"]
}

ZOOMS = ["0.5", "1.0"]
DILUTION = {
    "M00": 0,
    "M10": 10,
    "M20": 20,
    "M30": 30,
    "M40": 40
}

records = []

for batch, folders in DATASETS.items():
    for folder in folders:
        concentration = folder.replace("_B2", "")
        for zoom in ZOOMS:
            source = ROOT / folder / zoom

            if not source.exists():
                print("Missing folder:", source)
                continue

            for path in source.rglob("*"):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                records.append({
                    "batch": batch,
                    "concentration": concentration,
                    "dilution_percent": DILUTION[concentration],
                    "zoom": zoom,
                    "format": path.suffix.lower(),
                    "file_path": str(path.resolve())
                })

df = pd.DataFrame(records)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\n========== MANIFEST SUMMARY ==========")
print("Total images:", len(df))
print("\nBatch counts:")
print(df["batch"].value_counts().sort_index())

print("\nConcentration counts:")
print(df.groupby(["batch", "concentration"]).size())

print("\nZoom counts:")
print(df.groupby(["batch", "zoom"]).size())

print("\nFormat counts:")
print(df["format"].value_counts())

print("\nDuplicate paths:", df["file_path"].duplicated().sum())
print("\nSaved:", OUTPUT)
