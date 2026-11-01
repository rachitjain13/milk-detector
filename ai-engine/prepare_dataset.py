
from pathlib import Path
import json
import cv2

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

CATEGORIES = [
    "milk_glass",
    "empty_glass",
    "other_objects",
]

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}


def check_dataset():
    report = {
        "dataset_path": str(DATASET_DIR),
        "categories": {},
        "total_images": 0,
        "unreadable_images": [],
        "ready_for_training": False,
    }

    for category in CATEGORIES:
        folder = DATASET_DIR / category
        folder.mkdir(parents=True, exist_ok=True)

        files = sorted(
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        )

        readable_count = 0

        for file in files:
            image = cv2.imread(str(file))

            if image is None:
                report["unreadable_images"].append(str(file))
                continue

            height, width = image.shape[:2]

            if width < 64 or height < 64:
                report["unreadable_images"].append(str(file))
                continue

            readable_count += 1

        report["categories"][category] = {
            "image_count": len(files),
            "readable_count": readable_count,
        }

        report["total_images"] += readable_count

    # This only checks whether enough files exist to begin
    # an initial experiment. It does not certify dataset quality.
    enough_images = all(
        report["categories"][category]["readable_count"] >= 20
        for category in CATEGORIES
    )

    report["ready_for_training"] = (
        enough_images
        and not report["unreadable_images"]
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    check_dataset()
