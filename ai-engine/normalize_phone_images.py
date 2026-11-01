
from pathlib import Path
import pandas as pd
import cv2

manifest = pd.read_csv(
    "analysis_output/phone_rgb_batch_manifest.csv"
)

output_root = Path("research_data/phone_photo/normalized_2448")
output_root.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = (2448, 2448)

records = []
failed = []

for i, row in manifest.iterrows():
    source = Path(row["file_path"])

    image = cv2.imread(str(source))

    if image is None:
        failed.append(str(source))
        continue

    normalized = cv2.resize(
        image,
        TARGET_SIZE,
        interpolation=cv2.INTER_AREA
    )

    batch = row["batch"]
    concentration = row["concentration"]
    zoom = str(row["zoom"])

    destination_dir = (
        output_root / batch / concentration / zoom
    )
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / f"{i:05d}.png"

    success = cv2.imwrite(str(destination), normalized)

    if not success:
        failed.append(str(source))
        continue

    records.append({
        "batch": batch,
        "concentration": concentration,
        "dilution_percent": row["dilution_percent"],
        "zoom": row["zoom"],
        "file_path": str(destination)
    })

normalized_df = pd.DataFrame(records)

normalized_df.to_csv(
    "analysis_output/normalized_2448_manifest.csv",
    index=False
)

print("Original images:", len(manifest))
print("Normalized images:", len(normalized_df))
print("Failed:", len(failed))
print("Target resolution:", TARGET_SIZE)

if failed:
    print("Failed files:")
    for path in failed[:20]:
        print(path)
