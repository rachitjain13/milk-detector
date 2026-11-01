
from pathlib import Path
from PIL import Image

ROOT = Path("research_data/phone_photo/raw")

FOLDERS = [
    "M00_B2",
    "M10_B2",
    "M20_B2",
    "M30_B2",
    "M40_B2"
]

total = 0
corrupt = []

for folder in FOLDERS:
    jpg_files = list((ROOT / folder).rglob("*.jpg"))

    for path in jpg_files:
        try:
            with Image.open(path) as img:
                img.verify()
            total += 1

        except Exception as e:
            corrupt.append((str(path), str(e)))

    print(f"{folder}: {len(jpg_files)} JPG checked")

print("\n========== RESULT ==========")
print("Verified JPG:", total)
print("Corrupt JPG:", len(corrupt))

if corrupt:
    for path, error in corrupt:
        print(path, error)
