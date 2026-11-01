
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd

manifest = pd.read_csv(
    "analysis_output/spmilk_master_manifest.csv"
)

results = []

for index, row in manifest.iterrows():
    path = Path(row["file_path"])

    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode

        status = "OK"

    except Exception as e:
        width = height = None
        mode = None
        status = "CORRUPT"

    results.append({
        "file_path": str(path),
        "campaign": row["campaign"],
        "label": row["label"],
        "width": width,
        "height": height,
        "mode": mode,
        "status": status
    })

audit = pd.DataFrame(results)

output = "analysis_output/spmilk_image_audit.csv"
audit.to_csv(output, index=False)

print("Total checked:", len(audit))
print("\nImage status:")
print(audit["status"].value_counts())

print("\nImage dimensions:")
print(
    audit.groupby(["width", "height"]).size()
    .sort_values(ascending=False)
)

print("\nCampaign-wise corrupt images:")
print(
    audit[audit["status"] == "CORRUPT"]
    .groupby("campaign").size()
)

print("\nSaved:", output)
