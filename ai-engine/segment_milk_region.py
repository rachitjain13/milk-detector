
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ============================================================
# MILKOPTIC AI - AUTOMATIC MILK REGION SEGMENTATION
# Experiment 3
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = (
    BASE_DIR
    / "research_data"
    / "phone_photo"
    / "raw"
)

INPUT_CSV = (
    BASE_DIR
    / "analysis_output"
    / "camera_experiments"
    / "unified_all_batches.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "analysis_output"
    / "milk_segmentation_experiment"
)

MASK_DIR = OUTPUT_DIR / "masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
PREVIEW_DIR = OUTPUT_DIR / "previews"

FEATURE_CSV = OUTPUT_DIR / "segmented_milk_features.csv"
FAILURE_CSV = OUTPUT_DIR / "failed_images.csv"

for folder in [
    OUTPUT_DIR,
    MASK_DIR,
    OVERLAY_DIR,
    PREVIEW_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# First restrict processing to the centered area.
CENTER_WIDTH_RATIO = 0.42
CENTER_HEIGHT_RATIO = 0.42

# Candidate pixel conditions.
MIN_SATURATION = 0
MAX_SATURATION = 85
MIN_BRIGHTNESS = 80

# Minimum and maximum acceptable detected area
MIN_MASK_AREA_RATIO = 0.08
MAX_MASK_AREA_RATIO = 0.85

# Feature extraction resolution
RESIZE_SIZE = (256, 256)

# Save previews for manual verification
MAX_PREVIEWS = 40


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def resolve_image_path(row):

    batch = str(row["batch"]).strip().upper()
    relative_path = Path(str(row["relative_path"]))

    if batch == "B2":
        return RAW_DIR / relative_path

    concentration = str(row["concentration"]).strip()

    image_path = RAW_DIR / concentration / relative_path

    if image_path.exists():
        return image_path

    # Fallback for B1 images
    search_folder = RAW_DIR / concentration

    if search_folder.exists():

        matches = list(
            search_folder.rglob(relative_path.name)
        )

        if matches:

            zoom = str(row["zoom"]).strip()

            for match in matches:
                if zoom in match.parts:
                    return match

            return matches[0]

    return None


# ============================================================
# CENTRAL REGION
# ============================================================

def get_center_crop(image):

    height, width = image.shape[:2]

    crop_width = int(width * CENTER_WIDTH_RATIO)
    crop_height = int(height * CENTER_HEIGHT_RATIO)

    x1 = (width - crop_width) // 2
    y1 = (height - crop_height) // 2

    x2 = x1 + crop_width
    y2 = y1 + crop_height

    crop = image[y1:y2, x1:x2]

    return crop, (x1, y1, x2, y2)


# ============================================================
# CIRCULAR SPATIAL PRIOR
# ============================================================

def create_elliptical_prior(height, width):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    center = (width // 2, height // 2)

    axes = (
        max(1, int(width * 0.46)),
        max(1, int(height * 0.46))
    )

    cv2.ellipse(
        mask,
        center,
        axes,
        0,
        0,
        360,
        255,
        -1
    )

    return mask


# ============================================================
# AUTOMATIC MILK SEGMENTATION
# ============================================================

def segment_milk(crop):

    height, width = crop.shape[:2]

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    # --------------------------------------------------------
    # Central spatial prior
    # --------------------------------------------------------

    prior = create_elliptical_prior(
        height,
        width
    )

    # --------------------------------------------------------
    # Otsu brightness threshold
    # --------------------------------------------------------

    otsu_threshold, _ = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Do not use an excessively high threshold.
    adaptive_threshold = max(
        MIN_BRIGHTNESS,
        int(otsu_threshold * 0.75)
    )

    adaptive_threshold = min(
        adaptive_threshold,
        245
    )

    # --------------------------------------------------------
    # Candidate milk pixels
    # --------------------------------------------------------

    candidate = (
        (saturation >= MIN_SATURATION)
        &
        (saturation <= MAX_SATURATION)
        &
        (brightness >= adaptive_threshold)
        &
        (prior > 0)
    )

    mask = candidate.astype(np.uint8) * 255

    # --------------------------------------------------------
    # Morphological cleaning
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # --------------------------------------------------------
    # Keep largest connected component
    # --------------------------------------------------------

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if count <= 1:
        return None, {
            "status": "failed_no_component",
            "otsu_threshold": float(otsu_threshold),
            "adaptive_threshold": adaptive_threshold,
            "area_ratio": 0.0
        }

    component_areas = stats[1:, cv2.CC_STAT_AREA]

    largest_label = (
        int(np.argmax(component_areas)) + 1
    )

    largest_area = int(
        stats[largest_label, cv2.CC_STAT_AREA]
    )

    final_mask = (
        labels == largest_label
    ).astype(np.uint8) * 255

    area_ratio = largest_area / (
        height * width
    )

    # --------------------------------------------------------
    # Validate candidate mask
    # --------------------------------------------------------

    if area_ratio < MIN_MASK_AREA_RATIO:

        return None, {
            "status": "failed_mask_too_small",
            "otsu_threshold": float(otsu_threshold),
            "adaptive_threshold": adaptive_threshold,
            "area_ratio": float(area_ratio)
        }

    if area_ratio > MAX_MASK_AREA_RATIO:

        return None, {
            "status": "failed_mask_too_large",
            "otsu_threshold": float(otsu_threshold),
            "adaptive_threshold": adaptive_threshold,
            "area_ratio": float(area_ratio)
        }

    # --------------------------------------------------------
    # Segmentation bounding box
    # --------------------------------------------------------

    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])

    box_width = int(
        stats[largest_label, cv2.CC_STAT_WIDTH]
    )

    box_height = int(
        stats[largest_label, cv2.CC_STAT_HEIGHT]
    )

    return final_mask, {
        "status": "segmented",
        "otsu_threshold": float(otsu_threshold),
        "adaptive_threshold": adaptive_threshold,
        "area_ratio": float(area_ratio),
        "bbox_x": x,
        "bbox_y": y,
        "bbox_width": box_width,
        "bbox_height": box_height
    }


# ============================================================
# FEATURE EXTRACTION FROM MASKED PIXELS
# ============================================================

def extract_masked_features(crop, mask):

    features = {}

    resized_image = cv2.resize(
        crop,
        RESIZE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    resized_mask = cv2.resize(
        mask,
        RESIZE_SIZE,
        interpolation=cv2.INTER_NEAREST
    )

    valid = resized_mask > 0

    if np.count_nonzero(valid) < 50:
        raise ValueError("Not enough segmented pixels")

    # --------------------------------------------------------
    # RGB statistics
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2RGB
    )

    for index, channel_name in enumerate(
        ["red", "green", "blue"]
    ):

        values = rgb[:, :, index][valid].astype(
            np.float32
        )

        features[f"rgb_{channel_name}_mean"] = float(
            np.mean(values)
        )

        features[f"rgb_{channel_name}_std"] = float(
            np.std(values)
        )

        features[f"rgb_{channel_name}_median"] = float(
            np.median(values)
        )

        features[f"rgb_{channel_name}_p25"] = float(
            np.percentile(values, 25)
        )

        features[f"rgb_{channel_name}_p75"] = float(
            np.percentile(values, 75)
        )

    # --------------------------------------------------------
    # HSV statistics
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2HSV
    )

    for index, channel_name in enumerate(
        ["hue", "saturation", "value"]
    ):

        values = hsv[:, :, index][valid].astype(
            np.float32
        )

        features[f"hsv_{channel_name}_mean"] = float(
            np.mean(values)
        )

        features[f"hsv_{channel_name}_std"] = float(
            np.std(values)
        )

        features[f"hsv_{channel_name}_median"] = float(
            np.median(values)
        )

    # --------------------------------------------------------
    # Grayscale masked features
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        resized_image,
        cv2.COLOR_BGR2GRAY
    )

    gray_values = gray[valid].astype(np.float32)

    features["gray_mean"] = float(
        np.mean(gray_values)
    )

    features["gray_std"] = float(
        np.std(gray_values)
    )

    features["gray_median"] = float(
        np.median(gray_values)
    )

    features["gray_min"] = float(
        np.min(gray_values)
    )

    features["gray_max"] = float(
        np.max(gray_values)
    )

    # --------------------------------------------------------
    # Histogram: 16 bins
    # --------------------------------------------------------

    hist, _ = np.histogram(
        gray_values,
        bins=16,
        range=(0, 256)
    )

    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-8

    for i, value in enumerate(hist):
        features[f"gray_hist_{i}"] = float(value)

    # --------------------------------------------------------
    # Texture features
    # Fill non-segmented pixels with median intensity.
    # Texture is therefore calculated within the masked
    # bounding box, with the mask retained as a limitation.
    # --------------------------------------------------------

    masked_gray = gray.copy()

    median_value = int(
        np.median(gray_values)
    )

    masked_gray[~valid] = median_value

    # GLCM
    glcm_input = cv2.resize(
        masked_gray,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    glcm = graycomatrix(
        glcm_input,
        distances=[1, 3],
        angles=[0, np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True
    )

    for property_name in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]:

        values = graycoprops(
            glcm,
            property_name
        )

        features[f"glcm_{property_name}_mean"] = float(
            np.mean(values)
        )

        features[f"glcm_{property_name}_std"] = float(
            np.std(values)
        )

    # LBP
    lbp = local_binary_pattern(
        masked_gray,
        P=8,
        R=1,
        method="uniform"
    )

    lbp_values = lbp[valid]

    lbp_hist, _ = np.histogram(
        lbp_values,
        bins=np.arange(0, 11),
        range=(0, 10)
    )

    lbp_hist = lbp_hist.astype(np.float64)
    lbp_hist /= lbp_hist.sum() + 1e-8

    for i, value in enumerate(lbp_hist):
        features[f"lbp_{i}"] = float(value)

    # --------------------------------------------------------
    # Mask quality features
    # --------------------------------------------------------

    features["mask_pixel_count"] = int(
        np.count_nonzero(valid)
    )

    features["mask_area_ratio"] = float(
        np.count_nonzero(valid) / valid.size
    )

    return features


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(crop, mask):

    overlay = crop.copy()

    color = np.zeros_like(crop)
    color[:, :] = (0, 255, 0)

    selected = mask > 0

    overlay[selected] = cv2.addWeighted(
        crop[selected],
        0.55,
        color[selected],
        0.45,
        0
    )

    return overlay


# ============================================================
# MAIN PROCESS
# ============================================================

def main():

    print("\n==============================================")
    print(" MILKOPTIC AI - AUTOMATIC MILK SEGMENTATION")
    print("==============================================\n")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    if not RAW_DIR.exists():
        raise FileNotFoundError(RAW_DIR)

    df = pd.read_csv(INPUT_CSV)

    print(f"Total images: {len(df)}")
    print(f"B1 and B2 will be processed separately.")
    print("Segmentation method: HSV + brightness + spatial prior\n")

    results = []
    failures = []

    preview_count = 0
    segmented_count = 0
    rejected_count = 0

    for index, row in df.iterrows():

        image_path = resolve_image_path(row)

        try:

            if image_path is None:
                raise FileNotFoundError(
                    "Could not resolve image path"
                )

            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError(
                    "OpenCV could not read image"
                )

            # Center crop
            crop, crop_coordinates = get_center_crop(image)

            # Automatic segmentation
            mask, segmentation_info = segment_milk(crop)

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

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
                "segmentation_status": segmentation_info["status"],
                "otsu_threshold": segmentation_info["otsu_threshold"],
                "adaptive_threshold": segmentation_info["adaptive_threshold"],
                "segmentation_area_ratio": segmentation_info["area_ratio"]
            }

            # ------------------------------------------------
            # Successful segmentation
            # ------------------------------------------------

            if mask is not None:

                segmented_count += 1

                features = extract_masked_features(
                    crop,
                    mask
                )

                record.update(features)

                # Save mask and overlay
                filename = (
                    f"{index:05d}_"
                    f"{row['batch']}_"
                    f"{row['concentration']}_"
                    f"{row['zoom']}.png"
                )

                mask_path = MASK_DIR / filename

                cv2.imwrite(
                    str(mask_path),
                    mask
                )

                overlay = create_overlay(
                    crop,
                    mask
                )

                overlay_path = OVERLAY_DIR / filename

                cv2.imwrite(
                    str(overlay_path),
                    overlay
                )

                # Save a combined preview for initial images
                if preview_count < MAX_PREVIEWS:

                    original_preview = cv2.resize(
                        crop,
                        (320, 320)
                    )

                    mask_preview = cv2.resize(
                        cv2.cvtColor(
                            mask,
                            cv2.COLOR_GRAY2BGR
                        ),
                        (320, 320)
                    )

                    overlay_preview = cv2.resize(
                        overlay,
                        (320, 320)
                    )

                    combined = np.hstack([
                        original_preview,
                        mask_preview,
                        overlay_preview
                    ])

                    preview_path = (
                        PREVIEW_DIR
                        / f"preview_{preview_count + 1:02d}.jpg"
                    )

                    cv2.imwrite(
                        str(preview_path),
                        combined
                    )

                    preview_count += 1

            else:

                rejected_count += 1

            results.append(record)

        except Exception as error:

            failures.append({
                "row_index": index,
                "batch": row.get("batch"),
                "concentration": row.get("concentration"),
                "relative_path": row.get("relative_path"),
                "error": str(error)
            })

        if (index + 1) % 100 == 0:

            print(
                f"Processed: {index + 1}/{len(df)} | "
                f"Segmented: {segmented_count} | "
                f"Rejected: {rejected_count} | "
                f"Failed: {len(failures)}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    if results:

        output_df = pd.DataFrame(results)

        output_df.to_csv(
            FEATURE_CSV,
            index=False
        )

    if failures:

        pd.DataFrame(failures).to_csv(
            FAILURE_CSV,
            index=False
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n==============================================")
    print(" SEGMENTATION EXPERIMENT COMPLETED")
    print("==============================================")

    print(f"Total input images: {len(df)}")
    print(f"Processed rows: {len(results)}")
    print(f"Successfully segmented: {segmented_count}")
    print(f"Rejected masks: {rejected_count}")
    print(f"Failed images: {len(failures)}")
    print(f"Preview images: {preview_count}")

    if results:

        output_df = pd.DataFrame(results)

        print("\nSegmentation status:")

        print(
            output_df["segmentation_status"]
            .value_counts()
            .to_string()
        )

        print("\nBatch distribution:")

        print(
            output_df.groupby(
                ["batch", "segmentation_status"]
            ).size().to_string()
        )

        print("\nConcentration distribution:")

        print(
            output_df.groupby(
                ["batch", "concentration", "segmentation_status"]
            ).size().to_string()
        )

    print("\nFeature CSV:")
    print(FEATURE_CSV)

    print("\nMasks:")
    print(MASK_DIR)

    print("\nOverlays:")
    print(OVERLAY_DIR)

    print("\nPreviews:")
    print(PREVIEW_DIR)

    if failures:
        print("\nFailure report:")
        print(FAILURE_CSV)

    print("\n==============================================")


if __name__ == "__main__":
    main()
