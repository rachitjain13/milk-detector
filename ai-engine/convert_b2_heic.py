
from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

ROOT = Path("research_data/phone_photo/raw")

B2_FOLDERS = [
    "M00_B2",
    "M10_B2",
    "M20_B2",
    "M30_B2",
    "M40_B2"
]

converted = 0
failed = []

for folder in B2_FOLDERS:
    folder_path = ROOT / folder

    if not folder_path.exists():
        print("Folder missing:", folder)
        continue

    for heic_path in folder_path.rglob("*"):
        if not heic_path.is_file():
            continue

        if heic_path.suffix.lower() not in (".heic", ".heif"):
            continue

        jpg_path = heic_path.with_suffix(".jpg")

        try:
            with Image.open(heic_path) as img:
                img.convert("RGB").save(
                    jpg_path,
                    "JPEG",
                    quality=95
                )

            with Image.open(jpg_path) as check:
                check.verify()

            converted += 1
            print("Converted:", heic_path)

        except Exception as e:
            failed.append((str(heic_path), str(e)))
            print("FAILED:", heic_path, e)

print("\n========== SUMMARY ==========")
print("Successfully converted:", converted)
print("Failed:", len(failed))

if failed:
    print("\nFailed files:")
    for path, error in failed:
        print(path, error)
