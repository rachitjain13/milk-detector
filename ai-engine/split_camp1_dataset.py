import csv
import random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "research_data" / "sp_milk"
MANIFEST = DATA / "camp1_manifest.csv"
OUTPUT = DATA / "camp1_split.csv"

if not MANIFEST.exists():
    raise SystemExit(f"Manifest not found: {MANIFEST}")

with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise SystemExit("Manifest is empty.")

# Group all frames belonging to the same physical sample.
groups = defaultdict(list)
for row in rows:
    key = (row["label"].strip().lower(), row["sample"].strip())
    groups[key].append(row)

by_label = defaultdict(list)
for (label, sample), group_rows in groups.items():
    by_label[label].append((sample, group_rows))

rng = random.Random(42)
split_rows = []

for label, samples in sorted(by_label.items()):
    rng.shuffle(samples)

    if len(samples) < 3:
        raise SystemExit(
            f"Not enough independent samples for label '{label}': {len(samples)}"
        )

    # CAMP1 should have six samples per class: 4 train, 1 validation, 1 test.
    n = len(samples)
    n_val = max(1, round(n * 1 / 6))
    n_test = max(1, round(n * 1 / 6))
    n_train = n - n_val - n_test

    if n_train < 1:
        raise SystemExit(f"Cannot split {n} samples for label '{label}'.")

    for i, (sample, group_rows) in enumerate(samples):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "validation"
        else:
            split = "test"

        for row in group_rows:
            split_rows.append({
                **row,
                "split": split,
            })

fields = list(split_rows[0].keys())
with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(split_rows)

# Confirm that no sample appears in more than one split.
sample_splits = defaultdict(set)
for row in split_rows:
    sample_splits[(row["label"], row["sample"])].add(row["split"])

leaks = [key for key, values in sample_splits.items() if len(values) > 1]

print("Total images:", len(split_rows))
print("Unique samples:", len(sample_splits))
print("Images per label and split:")
for label in sorted({r["label"] for r in split_rows}):
    for split in ("train", "validation", "test"):
        count = sum(
            1 for r in split_rows
            if r["label"] == label and r["split"] == split
        )
        print(f"  {label:15} {split:10} {count}")

print("Sample leakage:", len(leaks))
print("Split CSV:", OUTPUT)

if leaks:
    raise SystemExit("ERROR: A sample appears in multiple splits.")
