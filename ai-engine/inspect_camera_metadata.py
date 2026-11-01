
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
import os

manifest = pd.read_csv(
    "analysis_output/phone_rgb_batch_manifest.csv"
)

selected = manifest[
    (manifest["batch"] == "B1") &
    (
        (manifest["concentration"] == "M20") |
        (manifest["concentration"] == "M00")
    )
]

print("\n===== IMAGE METADATA AUDIT =====")

for _, row in selected.iterrows():

    if row["camera_mode"] != "108MP":
        continue

    path = row["file_path"]

    try:
        img = Image.open(path)
        exif = img.getexif()

        print("\nFile:", os.path.basename(path))
        print("Dimensions:", img.size)

        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)

            if tag in [
                "Make",
                "Model",
                "DateTime",
                "FocalLength",
                "DigitalZoomRatio",
                "ExposureTime",
                "ISOSpeedRatings",
                "FNumber",
                "Software"
            ]:
                print(f"{tag}: {value}")

    except Exception as e:
        print("Metadata error:", path, e)
