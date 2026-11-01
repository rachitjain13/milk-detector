import pandas as pd
import cv2

manifest = pd.read_csv("analysis_output/phone_rgb_batch_manifest.csv")

print("\n===== UNUSUAL 8992 x 8992 IMAGES =====")

for _, row in manifest.iterrows():
    image = cv2.imread(row["file_path"])
    if image is None:
        continue

    height, width = image.shape[:2]

    if width == 8992 or height == 8992:
        print(
            f'{row["batch"]} | {row["concentration"]} | '
            f'{row["zoom"]} | {row["format"]} | {row["file_path"]}'
        )
