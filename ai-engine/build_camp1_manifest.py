import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "research_data" / "sp_milk"
EXTRACTED = DATA / "extracted"
LABELS_FILE = DATA / "filenames_and_labels.csv"
OUTPUT = DATA / "camp1_manifest.csv"

def find_image(row):
    category = row["Subdataset"].strip().lower()
    sample = row["Sample"].strip()
    filename = row["Filename"].strip()
    cuvette = row["Cuvette"].strip()

    if category == "adulterated":
        base = EXTRACTED / "CAMP1" / "CAMP1"
    elif category == "unadulterated":
        base = EXTRACTED / "CAMP1_UNADULTERATED" / "CAMP1"
    else:
        return None

    # Use the CSV sample and cuvette to resolve the exact path.
    path = base / sample / cuvette / filename
    return path if path.is_file() else None

if not LABELS_FILE.is_file():
    raise SystemExit(f"CSV not found: {LABELS_FILE}")

with LABELS_FILE.open("r", newline="", encoding="utf-8-sig") as f:
    rows = [
        row for row in csv.DictReader(f, delimiter=";")
        if row["Campaign"].strip() == "CAMP1"
    ]

manifest = []
missing = []
for row in rows:
    path = find_image(row)
    if path is None:
        missing.append(row)
        continue

    manifest.append({
        "image_path": str(path.resolve()),
        "label": row["Subdataset"].strip().lower(),
        "campaign": row["Campaign"].strip(),
        "sample": row["Sample"].strip(),
        "cuvette": row["Cuvette"].strip(),
        "filename": row["Filename"].strip(),
    })

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_path", "label", "campaign",
            "sample", "cuvette", "filename"
        ]
    )
    writer.writeheader()
    writer.writerows(manifest)

print(f"CSV records for CAMP1: {len(rows)}")
print(f"Images matched: {len(manifest)}")
print(f"Images missing: {len(missing)}")
print(f"Manifest saved: {OUTPUT}")
print("Matched labels:", dict(Counter(r["label"] for r in manifest)))

if missing:
    print("\nFirst 10 missing records:")
    for row in missing[:10]:
        print(
            row.get("Subdataset"), row.get("Sample"),
            row.get("Cuvette"), row.get("Filename")
        )
