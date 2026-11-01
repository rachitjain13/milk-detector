
import pandas as pd
import cv2
import os

manifest = pd.read_csv(
    "analysis_output/phone_rgb_batch_manifest.csv"
)

selected = manifest[
    (manifest["batch"] == "B1") &
    (manifest["concentration"] == "M20")
]

results = []

for _, row in selected.iterrows():

    path = row["file_path"]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        continue

    height, width = img.shape[:2]

    results.append({
        "camera_mode": row["camera_mode"],
        "zoom": row["zoom"],
        "format": row["format"],
        "width": width,
        "height": height,
        "shape": str(img.shape),
        "dtype": str(img.dtype),
        "channels": 1 if len(img.shape) == 2 else img.shape[2],
        "file_size_kb": round(os.path.getsize(path) / 1024, 2),
        "mean_pixel_value": round(float(img.mean()), 3),
        "std_pixel_value": round(float(img.std()), 3)
    })

df = pd.DataFrame(results)

print("\n===== CAMERA IMAGE PROPERTIES =====")
print(
    df.groupby(["camera_mode", "zoom"]).agg({
        "width": "first",
        "height": "first",
        "dtype": "first",
        "channels": "first",
        "file_size_kb": ["mean", "min", "max"],
        "mean_pixel_value": "mean",
        "std_pixel_value": "mean",
        "camera_mode": "count"
    }).to_string()
)

df.to_csv(
    "analysis_output/camera_image_properties.csv",
    index=False
)

print("\nSaved camera image properties.")
