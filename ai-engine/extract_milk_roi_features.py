
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# ============================================================
# MILKOPTIC AI - MILK REGION (ROI) FEATURE EXTRACTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    BASE_DIR
    / "analysis_output"
    / "camera_experiments"
    / "unified_all_batches.csv"
)

RAW_DIR = (
    BASE_DIR
    / "research_data"
    / "phone_photo"
    / "raw"
)

OUTPUT_DIR = (
    BASE_DIR
    / "analysis_output"
    / "milk_roi_experiment"
)

PREVIEW_DIR = OUTPUT_DIR / "previews"

OUTPUT_CSV = OUTPUT_DIR / "milk_roi_features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Centered ROI: 42% of image width and height
ROI_WIDTH_RATIO = 0.42
ROI_HEIGHT_RATIO = 0.42

# Resize ROI for consistent feature extraction
RESIZE_SIZE = (256, 256)

# Save previews for visual verification
MAX_PREVIEWS = 20


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def resolve_image_path(row):
    """
    Resolve original image path for B1 and B2.

    B1:
    raw/M00/0.5/jpg/image.jpg

    B2:
    raw/M00_B2/0.5/image.jpg
    """

    batch = str(row["batch"]).strip()
    concentration = str(row["concentration"]).strip()

    relative_path = Path(str(row["relative_path"]))

    if batch.upper() == "B2":
        folder_name = f"{concentration}_B2"
    else:
        folder_name = concentration

    image_path = RAW_DIR / folder_name / relative_path

    if image_path.exists():
        return image_path

    # Fallback: search for the filename inside its
    # batch/concentration directory.
    search_folder = RAW_DIR / folder_name
    filename = relative_path.name

    if search_folder.exists():
        matches = list(search_folder.rglob(filename))

        if matches:
            zoom = str(row["zoom"]).strip()

            # Prefer a match from the correct zoom folder.
            for match in matches:
                if zoom in match.parts:
                    return match

            return matches[0]

    return None


# ============================================================
# CENTERED MILK ROI
# ============================================================

def extract_milk_roi(image):

    h, w = image.shape[:2]

    crop_w = int(w * ROI_WIDTH_RATIO)
    crop_h = int(h * ROI_HEIGHT_RATIO)

    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2

    x2 = x1 + crop_w
    y2 = y1 + crop_h

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return None, None

    coordinates = (x1, y1, x2, y2)

    return roi, coordinates


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(roi):

    features = {}

    # Resize ROI
    roi = cv2.resize(
        roi,
        RESIZE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # GRAYSCALE FEATURES
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    features["gray_mean"] = float(np.mean(gray))
    features["gray_std"] = float(np.std(gray))
    features["gray_median"] = float(np.median(gray))
    features["gray_min"] = float(np.min(gray))
    features["gray_max"] = float(np.max(gray))

    # Sharpness
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    features["sharpness"] = float(laplacian.var())

    # Edge density
    edges = cv2.Canny(gray, 50, 150)

    features["edge_density"] = float(
        np.count_nonzero(edges) / edges.size
    )

    # Grayscale histogram - 16 bins
    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [16],
        [0, 256]
    ).flatten()

    hist = hist / (hist.sum() + 1e-8)

    for i, value in enumerate(hist):
        features[f"gray_hist_{i}"] = float(value)

    # --------------------------------------------------------
    # RGB FEATURES
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    channel_names = ["red", "green", "blue"]

    for i, channel_name in enumerate(channel_names):

        channel = rgb[:, :, i]

        features[f"{channel_name}_mean"] = float(
            np.mean(channel)
        )

        features[f"{channel_name}_std"] = float(
            np.std(channel)
        )

        features[f"{channel_name}_median"] = float(
            np.median(channel)
        )

    # --------------------------------------------------------
    # HSV FEATURES
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    hsv_names = ["hue", "saturation", "value"]

    for i, channel_name in enumerate(hsv_names):

        channel = hsv[:, :, i]

        features[f"{channel_name}_mean"] = float(
            np.mean(channel)
        )

        features[f"{channel_name}_std"] = float(
            np.std(channel)
        )

        features[f"{channel_name}_median"] = float(
            np.median(channel)
        )

    return features


# ============================================================
# MAIN PROCESS
# ============================================================

def main():

    print("\n========================================")
    print(" MILKOPTIC AI - MILK ROI EXPERIMENT")
    print("========================================\n")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset folder not found: {RAW_DIR}"
        )

    df = pd.read_csv(INPUT_CSV)

    print(f"Total manifest rows: {len(df)}")
    print(f"Input file: {INPUT_CSV}")
    print(f"ROI size: {ROI_WIDTH_RATIO:.0%} x "
          f"{ROI_HEIGHT_RATIO:.0%}\n")

    results = []

    processed = 0
    failed = 0
    preview_count = 0

    missing_paths = []

    for index, row in df.iterrows():

        image_path = resolve_image_path(row)

        if image_path is None:

            failed += 1

            if len(missing_paths) < 20:
                missing_paths.append(
                    str(row["relative_path"])
                )

            continue

        image = cv2.imread(str(image_path))

        if image is None:

            failed += 1
            continue

        # Extract centered ROI
        roi, coordinates = extract_milk_roi(image)

        if roi is None:
            failed += 1
            continue

        # Extract numerical features
        features = extract_features(roi)

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        record = {
            "batch": row["batch"],
            "concentration": row["concentration"],
            "dilution_percent": row["dilution_percent"],
            "zoom": row["zoom"],
            "file_name": image_path.name,
            "relative_path": row["relative_path"],
            "camera_mode": row.get("camera_mode", ""),
            "image_width_original": image.shape[1],
            "image_height_original": image.shape[0],
        }

        record.update(features)

        results.append(record)

        processed += 1

        # ----------------------------------------------------
        # SAVE ROI PREVIEWS
        # ----------------------------------------------------

        if preview_count < MAX_PREVIEWS:

            x1, y1, x2, y2 = coordinates

            preview = image.copy()

            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                8
            )

            preview_name = (
                f"preview_{preview_count + 1:02d}_"
                f"{image_path.stem}.jpg"
            )

            preview_path = PREVIEW_DIR / preview_name

            cv2.imwrite(
                str(preview_path),
                preview
            )

            preview_count += 1

        # Progress
        if processed % 100 == 0:
            print(
                f"Processed: {processed}/{len(df)}"
            )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    if not results:
        print("\nNo images were successfully processed.")
        print("Check the image paths and input CSV.")
        return

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print(" EXTRACTION COMPLETED")
    print("========================================")

    print(f"Manifest rows:       {len(df)}")
    print(f"Processed images:    {processed}")
    print(f"Failed images:       {failed}")
    print(f"Feature columns:     {len(output_df.columns)}")
    print(f"Preview images:      {preview_count}")

    print("\nBatch distribution:")
    print(output_df["batch"].value_counts().to_string())

    print("\nConcentration distribution:")
    print(
        output_df["concentration"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nOutput CSV:")
    print(OUTPUT_CSV)

    print("\nPreview folder:")
    print(PREVIEW_DIR)

    if missing_paths:
        print("\nFirst missing relative paths:")
        for path in missing_paths[:10]:
            print(" -", path)

    print("\n========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
