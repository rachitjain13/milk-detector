
#!/usr/bin/env python
"""
MilkOptic AI - 20 Image ROI Validation

Selects one representative image from every:
- Batch: B1, B2
- Concentration: M00, M10, M20, M30, M40
- Zoom: 0.5, 1.0

Generates:
1. Selected image manifest
2. Candidate detection previews (green)
3. Fixed milk-body ROI previews (orange)
4. Two comparison contact sheets

Original images and models are never modified.
"""

from pathlib import Path
import csv
import cv2
import numpy as np

from analyze_image import detect_candidate_region


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_ROOT = Path(
    "research_data/phone_photo/normalized_2448"
)

OUTPUT_ROOT = Path(
    "analysis_output/roi_validation_20"
)

BATCHES = ["B1", "B2"]

CONCENTRATIONS = [
    "M00", "M10", "M20", "M30", "M40"
]

ZOOMS = ["0.5", "1.0"]

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".webp"
}

THUMB_SIZE = 480


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

CANDIDATE_DIR = OUTPUT_ROOT / "candidate_previews"
MILK_ROI_DIR = OUTPUT_ROOT / "milk_roi_previews"

CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
MILK_ROI_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SELECT MIDDLE IMAGE FROM EACH GROUP
# ============================================================

selected = []

print("=" * 65)
print("MILKOPTIC AI - 20 IMAGE ROI VALIDATION")
print("=" * 65)

for batch in BATCHES:

    for concentration in CONCENTRATIONS:

        for zoom in ZOOMS:

            folder = (
                INPUT_ROOT /
                batch /
                concentration /
                zoom
            )

            if not folder.exists():
                print("MISSING FOLDER:", folder)
                continue

            images = sorted(
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            if not images:
                print("NO IMAGES:", folder)
                continue

            middle_index = len(images) // 2
            selected_image = images[middle_index]

            selected.append({
                "batch": batch,
                "concentration": concentration,
                "zoom": zoom,
                "source": selected_image
            })

            print(
                f"Selected: {batch} | "
                f"{concentration} | "
                f"{zoom}x | "
                f"{selected_image.name}"
            )


# ============================================================
# GENERATE PREVIEWS
# ============================================================

manifest_rows = []
candidate_tiles = []
milk_roi_tiles = []

for index, item in enumerate(selected, start=1):

    source = item["source"]

    image = cv2.imread(str(source))

    if image is None:
        print("READ FAILED:", source)
        continue

    height, width = image.shape[:2]

    candidate = detect_candidate_region(image)

    candidate_preview = image.copy()
    milk_preview = image.copy()

    # --------------------------------------------------------
    # GREEN: Candidate detection rectangle
    # --------------------------------------------------------

    if candidate is not None:

        x = candidate["x"]
        y = candidate["y"]
        w = candidate["width"]
        h = candidate["height"]

        cv2.rectangle(
            candidate_preview,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            6
        )

        candidate_status = "candidate_found"

    else:

        candidate_status = "candidate_not_found"

    # --------------------------------------------------------
    # ORANGE: Existing fixed milk-body ROI
    # --------------------------------------------------------

    mx1 = int(width * 0.29)
    mx2 = int(width * 0.71)

    my1 = int(height * 0.29)
    my2 = int(height * 0.79)

    cv2.rectangle(
        milk_preview,
        (mx1, my1),
        (mx2, my2),
        (0, 165, 255),
        6
    )

    cv2.putText(
        milk_preview,
        "ESTIMATED ROI - NOT VERIFIED",
        (max(10, mx1), max(35, my1 - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 100, 255),
        3,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # SAVE INDIVIDUAL PREVIEWS
    # --------------------------------------------------------

    label = (
        f"{index:02d}_{item['batch']}_"
        f"{item['concentration']}_Z{item['zoom']}"
    )

    candidate_path = CANDIDATE_DIR / f"{label}.png"
    milk_path = MILK_ROI_DIR / f"{label}.png"

    cv2.imwrite(str(candidate_path), candidate_preview)
    cv2.imwrite(str(milk_path), milk_preview)

    # --------------------------------------------------------
    # CREATE CONTACT SHEET THUMBNAILS
    # --------------------------------------------------------

    def make_tile(img, title):

        thumb = cv2.resize(
            img,
            (THUMB_SIZE, THUMB_SIZE),
            interpolation=cv2.INTER_AREA
        )

        header = np.full(
            (55, THUMB_SIZE, 3),
            245,
            dtype=np.uint8
        )

        cv2.putText(
            header,
            title,
            (10, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (20, 20, 20),
            2,
            cv2.LINE_AA
        )

        return np.vstack([header, thumb])

    group_label = (
        f"{item['batch']} {item['concentration']} "
        f"{item['zoom']}x"
    )

    candidate_tiles.append(
        make_tile(candidate_preview, group_label)
    )

    milk_roi_tiles.append(
        make_tile(milk_preview, group_label)
    )

    manifest_rows.append({
        "batch": item["batch"],
        "concentration": item["concentration"],
        "zoom": item["zoom"],
        "filename": source.name,
        "source_path": str(source),
        "candidate_status": candidate_status,
        "candidate_preview": str(candidate_path),
        "milk_roi_preview": str(milk_path)
    })


# ============================================================
# BUILD CONTACT SHEETS
# ============================================================

def build_contact_sheet(tiles, output_path):

    if not tiles:
        return

    columns = 4
    rows = []

    for start in range(0, len(tiles), columns):

        group = tiles[start:start + columns]

        while len(group) < columns:
            group.append(
                np.full(
                    (THUMB_SIZE + 55, THUMB_SIZE, 3),
                    245,
                    dtype=np.uint8
                )
            )

        rows.append(np.hstack(group))

    sheet = np.vstack(rows)

    cv2.imwrite(str(output_path), sheet)


build_contact_sheet(
    candidate_tiles,
    OUTPUT_ROOT / "candidate_contact_sheet.png"
)

build_contact_sheet(
    milk_roi_tiles,
    OUTPUT_ROOT / "milk_roi_contact_sheet.png"
)


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_path = OUTPUT_ROOT / "roi_validation_manifest.csv"

if manifest_rows:

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=manifest_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(manifest_rows)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ROI VALIDATION SUMMARY")
print("=" * 65)

print("Expected images:", 20)
print("Selected images:", len(selected))
print("Successfully processed:", len(manifest_rows))

print("\nBATCH DISTRIBUTION")

for batch in BATCHES:

    count = sum(
        row["batch"] == batch
        for row in manifest_rows
    )

    print(batch, ":", count)

print("\nOUTPUT FILES")

print("Candidate sheet:")
print(OUTPUT_ROOT / "candidate_contact_sheet.png")

print("\nMilk ROI sheet:")
print(OUTPUT_ROOT / "milk_roi_contact_sheet.png")

print("\nManifest:")
print(manifest_path)

print("\nROI VALIDATION PREPARATION COMPLETED")
print("=" * 65)
