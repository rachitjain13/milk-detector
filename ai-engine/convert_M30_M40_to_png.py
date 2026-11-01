
from pathlib import Path
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

base = Path("research_data/phone_photo/raw")

for sample in ["M30", "M40"]:
    print(f"\nProcessing {sample}")

    for zoom in ["0.5", "1.0"]:
        source = base / sample / zoom
        destination = source / "png"
        destination.mkdir(parents=True, exist_ok=True)

        if not source.exists():
            print(f"{zoom}: folder not found - {source}")
            continue

        converted = 0
        failed = 0

        for file in source.iterdir():
            if file.suffix.lower() in [".heic", ".heif"]:
                try:
                    with Image.open(file) as img:
                        img.convert("RGB").save(
                            destination / f"{file.stem}.png"
                        )
                    converted += 1
                except Exception as e:
                    failed += 1
                    print("Failed:", file.name, e)

        print(
            f"{sample} | {zoom}x: "
            f"converted={converted}, failed={failed}"
        )

print("\nConversion process completed.")
