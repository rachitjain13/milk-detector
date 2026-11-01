from pathlib import Path
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

base = Path("research_data/phone_photo/raw/M20")

for zoom in ["0.5", "1.0"]:
    source = base / zoom
    destination = source / "png"
    destination.mkdir(parents=True, exist_ok=True)

    count = 0
    failed = 0

    for file in source.iterdir():
        if file.suffix.lower() in [".heic", ".heif"]:
            try:
                with Image.open(file) as img:
                    img.convert("RGB").save(
                        destination / f"{file.stem}.png"
                    )
                count += 1
            except Exception as e:
                failed += 1
                print("Failed:", file.name, e)

    print(f"{zoom}: converted={count}, failed={failed}")
    