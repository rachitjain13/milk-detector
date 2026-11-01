
from pathlib import Path
from PIL import Image

ROOT = Path("research_data/phone_photo/raw")
DATASETS = ["M00", "M10", "M20", "M30", "M40"]
ZOOMS = ["0.5", "1.0"]

verified = 0
missing = []
corrupt = []

for dataset in DATASETS:
    for zoom in ZOOMS:
        source = ROOT / dataset / zoom
        output_folder = source / ("jpg" if dataset == "M00" else "png")
        output_ext = ".jpg" if dataset == "M00" else ".png"

        for heic in source.glob("*"):
            if heic.suffix.lower() not in (".heic", ".heif"):
                continue

            converted = output_folder / (heic.stem + output_ext)

            if not converted.exists():
                missing.append(str(heic))
                continue

            try:
                with Image.open(converted) as img:
                    img.verify()
                verified += 1
            except Exception as e:
                corrupt.append((str(converted), str(e)))

print("Verified converted images:", verified)
print("Missing conversions:", len(missing))
print("Corrupt conversions:", len(corrupt))

for item in missing:
    print("MISSING:", item)

for item in corrupt:
    print("CORRUPT:", item)
