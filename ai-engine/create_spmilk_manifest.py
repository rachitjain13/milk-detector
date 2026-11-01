
from pathlib import Path
import pandas as pd
import re

BASE = Path("research_data/sp_milk/extracted")

extensions = {".png", ".jpg", ".jpeg", ".bmp"}

rows = []

for folder in sorted(BASE.iterdir()):
    if not folder.is_dir():
        continue

    is_unadulterated = folder.name.endswith("_UNADULTERATED")

    campaign = folder.name.replace("_UNADULTERATED", "")
    label = 0 if is_unadulterated else 1

    for image in folder.rglob("*"):
        if not image.is_file() or image.suffix.lower() not in extensions:
            continue

        relative = image.relative_to(folder)
        parts = relative.parts

        category = parts[0] if len(parts) > 1 else "UNKNOWN"

        match = re.search(r"(JUG\d+)", image.name, re.IGNORECASE)
        jug = match.group(1).upper() if match else "UNKNOWN"

        rows.append({
            "file_path": str(image),
            "campaign": campaign,
            "category": category,
            "jug": jug,
            "label": label,
            "class_name": (
                "UNADULTERATED" if label == 0 else "ADULTERATED"
            )
        })

df = pd.DataFrame(rows)

output = Path("analysis_output/spmilk_master_manifest.csv")
output.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output, index=False)

print("Total images:", len(df))
print("\nClass counts:")
print(df["class_name"].value_counts())

print("\nCampaign counts:")
print(pd.crosstab(df["campaign"], df["class_name"]))

print("\nUnknown jug identifiers:", (df["jug"] == "UNKNOWN").sum())

print("\nManifest saved:", output)
