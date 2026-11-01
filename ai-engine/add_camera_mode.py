import pandas as pd
import cv2

manifest_path = "analysis_output/phone_rgb_batch_manifest.csv"

df = pd.read_csv(manifest_path)

df["camera_mode"] = "12MP"

for idx, row in df.iterrows():
    if row["batch"] == "B1":
        image = cv2.imread(row["file_path"])

        if image is not None:
            h, w = image.shape[:2]

            if w == 8992 and h == 8992:
                df.loc[idx, "camera_mode"] = "108MP"

df.to_csv(manifest_path, index=False)

print("\n===== CAMERA MODE SUMMARY =====")
print(pd.crosstab(df["batch"], df["camera_mode"]))

print("\n===== CONCENTRATION SUMMARY =====")
print(pd.crosstab(df["concentration"], df["camera_mode"]))

print("\nUpdated manifest:", manifest_path)
