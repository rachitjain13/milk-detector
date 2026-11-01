from pathlib import Path
import pandas as pd
import cv2

manifest = pd.read_csv(
    "analysis_output/phone_rgb_batch_manifest.csv"
)

rows = []

for _, item in manifest.iterrows():
    path = Path(item["file_path"])
    image = cv2.imread(str(path))

    if image is None:
        continue

    h, w = image.shape[:2]

    rows.append({
        "batch": item["batch"],
        "concentration": item["concentration"],
        "zoom": item["zoom"],
        "format": item["format"],
        "width": w,
        "height": h,
        "file_size_kb": path.stat().st_size / 1024
    })

df = pd.DataFrame(rows)

print("Image count:", len(df))
print("\n===== FORMAT SUMMARY =====")
print(
    df.groupby(["batch", "format"])[
        ["width", "height", "file_size_kb"]
    ].agg(["mean", "min", "max"]).round(2).to_string()
)

print("\n===== DIMENSION COUNTS =====")
print(df.groupby(["batch", "width", "height"]).size().to_string())

df.to_csv(
    "analysis_output/image_format_audit.csv",
    index=False
)

print("\nSaved: analysis_output/image_format_audit.csv")
