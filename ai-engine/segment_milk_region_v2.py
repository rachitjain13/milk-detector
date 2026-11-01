from pathlib import Path
import gc
import re
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ============================================================
# MILKOPTIC AI - AUTOMATIC MILK SEGMENTATION V2
# Complete, memory-efficient replacement
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "research_data" / "phone_photo" / "raw"
INPUT_CSV = BASE_DIR / "analysis_output" / "camera_experiments" / "unified_all_batches.csv"

OUTPUT_DIR = BASE_DIR / "analysis_output" / "milk_segmentation_v2"
MASK_DIR = OUTPUT_DIR / "masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
PREVIEW_DIR = OUTPUT_DIR / "previews"

FEATURE_OUTPUT = OUTPUT_DIR / "segmented_milk_features.csv"
FAILURE_OUTPUT = OUTPUT_DIR / "failed_images.csv"
PROGRESS_OUTPUT = OUTPUT_DIR / "progress_features.csv"

for folder in (OUTPUT_DIR, MASK_DIR, OVERLAY_DIR, PREVIEW_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ---------------- SETTINGS ----------------
CROP_RATIO = 0.42
PROCESS_SIZE = 640              # Resize before processing to control RAM
MIN_RADIUS_RATIO = 0.34
MAX_RADIUS_RATIO = 0.48
EXPECTED_RADIUS_RATIO = 0.43
RADIUS_SHRINK = 0.96
MAX_SATURATION = 120
MIN_MASK_AREA_RATIO = 0.10
MAX_MASK_AREA_RATIO = 0.75
PREVIEWS_PER_GROUP = 2
CHECKPOINT_EVERY = 100


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def resolve_image_path(row):
    batch = str(row["batch"])
    concentration = str(row["concentration"])
    relative_path = str(row["relative_path"]).replace("\\", "/")

    if batch == "B1":
        path = RAW_DIR / concentration / relative_path
    else:
        path = RAW_DIR / relative_path

    if path.is_file():
        return path

    filename = Path(relative_path).name
    matches = list(RAW_DIR.rglob(filename))
    return matches[0] if matches else None


def make_processing_crop(image):
    h, w = image.shape[:2]
    crop_w = max(1, int(w * CROP_RATIO))
    crop_h = max(1, int(h * CROP_RATIO))
    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    crop = image[y1:y1 + crop_h, x1:x1 + crop_w]

    if crop.size == 0:
        raise ValueError("Central crop is empty")

    interpolation = cv2.INTER_AREA if max(crop.shape[:2]) > PROCESS_SIZE else cv2.INTER_LINEAR
    return cv2.resize(crop, (PROCESS_SIZE, PROCESS_SIZE), interpolation=interpolation)


def detect_liquid_circle(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    min_dim = min(h, w)
    cx, cy = w // 2, h // 2
    expected_radius = int(min_dim * EXPECTED_RADIUS_RATIO)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1, min_dim // 3),
        param1=100,
        param2=35,
        minRadius=int(min_dim * MIN_RADIUS_RATIO),
        maxRadius=int(min_dim * MAX_RADIUS_RATIO),
    )

    if circles is None:
        return cx, cy, int(expected_radius * RADIUS_SHRINK), "center_circle_fallback"

    def score(circle):
        x, y, radius = circle
        return np.hypot(x - cx, y - cy) + abs(radius - expected_radius)

    selected = min(circles[0], key=score)
    x, y, radius = int(round(selected[0])), int(round(selected[1])), int(round(selected[2] * RADIUS_SHRINK))

    if np.hypot(x - cx, y - cy) > min_dim * 0.12:
        return cx, cy, int(expected_radius * RADIUS_SHRINK), "center_circle_fallback"

    return x, y, radius, "hough_circle"


def create_milk_mask(crop):
    h, w = crop.shape[:2]
    cx, cy, radius, circle_method = detect_liquid_circle(crop)

    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), radius, 255, -1)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    color_mask = np.where(saturation <= MAX_SATURATION, 255, 0).astype(np.uint8)

    # No strict brightness threshold: preserve darker milk regions.
    mask = cv2.bitwise_and(circle_mask, color_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        center_label = int(labels[cy, cx])
        if center_label > 0:
            mask = np.where(labels == center_label, 255, 0).astype(np.uint8)
        else:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    area_ratio = cv2.countNonZero(mask) / float(h * w)
    if area_ratio < MIN_MASK_AREA_RATIO or area_ratio > MAX_MASK_AREA_RATIO:
        return None, {
            "status": "rejected_area",
            "area_ratio": area_ratio,
            "circle_method": circle_method,
        }

    return mask, {
        "status": "segmented",
        "area_ratio": area_ratio,
        "circle_method": circle_method,
    }


def extract_features(crop, mask):
    features = {}
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    valid = mask > 0
    rgb_pixels = rgb[valid].astype(np.float32)
    hsv_pixels = hsv[valid].astype(np.float32)
    gray_pixels = gray[valid].astype(np.uint8)

    if gray_pixels.size == 0:
        raise ValueError("Segmentation mask contains no pixels")

    for idx, channel in enumerate(("R", "G", "B")):
        values = rgb_pixels[:, idx]
        for stat, value in (
            ("mean", np.mean(values)),
            ("std", np.std(values)),
            ("median", np.median(values)),
            ("min", np.min(values)),
            ("max", np.max(values)),
        ):
            features[f"{channel}_{stat}"] = float(value)

    for idx, channel in enumerate(("H", "S", "V")):
        values = hsv_pixels[:, idx]
        for stat, value in (
            ("mean", np.mean(values)),
            ("std", np.std(values)),
            ("median", np.median(values)),
            ("min", np.min(values)),
            ("max", np.max(values)),
        ):
            features[f"{channel}_{stat}"] = float(value)

    features["gray_mean"] = float(np.mean(gray_pixels))
    features["gray_std"] = float(np.std(gray_pixels))
    features["gray_median"] = float(np.median(gray_pixels))

    hist = cv2.calcHist([gray], [0], mask, [16], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    for i, value in enumerate(hist):
        features[f"gray_hist_{i}"] = float(value)

    # Texture is bounded to 256x256. Pixels outside mask are filled
    # with the median masked intensity; boundary texture may be affected.
    texture = gray.copy()
    texture[~valid] = int(np.median(gray_pixels))
    texture = cv2.resize(texture, (256, 256), interpolation=cv2.INTER_AREA)
    quantized = (texture // 16).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True,
    )

    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        values = graycoprops(glcm, prop)
        features[f"glcm_{prop}_mean"] = float(np.mean(values))
        features[f"glcm_{prop}_std"] = float(np.std(values))

    lbp = local_binary_pattern(texture, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)
    for i, value in enumerate(lbp_hist):
        features[f"lbp_{i}"] = float(value)

    features["mask_area_ratio"] = float(np.mean(valid))
    return features


def make_overlay(crop, mask):
    green = np.zeros_like(crop)
    green[:, :] = (0, 255, 0)
    return np.where(
        mask[:, :, None] > 0,
        cv2.addWeighted(crop, 0.45, green, 0.55, 0),
        crop,
    )


def save_preview(crop, mask, output_path):
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    panel = np.hstack((crop, mask_display, make_overlay(crop, mask)))
    cv2.imwrite(str(output_path), panel)


def save_checkpoint(feature_rows, failed_rows):
    if feature_rows:
        pd.DataFrame(feature_rows).to_csv(PROGRESS_OUTPUT, index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(FAILURE_OUTPUT, index=False)


def main():
    print("\n==============================================")
    print(" MILKOPTIC AI - SEGMENTATION V2")
    print(" MEMORY-EFFICIENT PROCESSING")
    print("==============================================")

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input manifest not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"batch", "concentration", "relative_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    print("Total images:", len(df))
    print(f"Processing crop size: {PROCESS_SIZE} x {PROCESS_SIZE}")

    feature_rows = []
    failed_rows = []
    preview_counts = {}

    segmented_count = 0
    rejected_count = 0
    processing_failed_count = 0

    for index, row in df.iterrows():
        image_path = None
        crop = None
        mask = None

        try:
            image_path = resolve_image_path(row)
            if image_path is None:
                raise FileNotFoundError(f"Image not found: {row['relative_path']}")

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cv2.imread returned None")

            crop = make_processing_crop(image)
            del image

            mask, info = create_milk_mask(crop)

            if mask is None:
                rejected_count += 1
                failed_rows.append({
                    **row.to_dict(),
                    "error": info["status"],
                    "area_ratio": info["area_ratio"],
                    "failure_type": "rejected_mask",
                })
                continue

            features = extract_features(crop, mask)
            segmented_count += 1

            rel = str(row["relative_path"])
            unique_name = (
                f"{index}_{safe_name(row['batch'])}_"
                f"{safe_name(row['concentration'])}_"
                f"{safe_name(row.get('zoom', 'unknown'))}_"
                f"{safe_name(Path(rel).stem)}"
            )

            cv2.imwrite(str(MASK_DIR / f"{unique_name}.png"), mask)
            cv2.imwrite(
                str(OVERLAY_DIR / f"{unique_name}.jpg"),
                make_overlay(crop, mask),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            group_key = (
                str(row["batch"]),
                str(row["concentration"]),
                str(row.get("zoom", "unknown")),
            )
            preview_counts[group_key] = preview_counts.get(group_key, 0)
            if preview_counts[group_key] < PREVIEWS_PER_GROUP:
                save_preview(
                    crop,
                    mask,
                    PREVIEW_DIR / f"{unique_name}_preview.jpg",
                )
                preview_counts[group_key] += 1

            feature_rows.append({
                **row.to_dict(),
                **features,
                "segmentation_status": "segmented",
                "circle_method": info["circle_method"],
                "mask_area_ratio": info["area_ratio"],
            })

        except Exception as error:
            processing_failed_count += 1
            failed_rows.append({
                **row.to_dict(),
                "error": repr(error),
                "failure_type": "processing_exception",
                "image_path": str(image_path) if image_path else "",
            })
            if processing_failed_count <= 20:
                print(
                    f"\nIMAGE ERROR at row {index}: "
                    f"{row.get('relative_path', 'unknown')}\n"
                    f"Reason: {repr(error)}\n"
                )

        finally:
            del crop, mask
            if (index + 1) % 25 == 0:
                gc.collect()

        if (index + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(feature_rows, failed_rows)
            print(
                f"Processed: {index + 1}/{len(df)} | "
                f"Segmented: {segmented_count} | "
                f"Rejected: {rejected_count} | "
                f"Failed: {processing_failed_count}"
            )

    # Final output files
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(FEATURE_OUTPUT, index=False)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(FAILURE_OUTPUT, index=False)
    elif FAILURE_OUTPUT.exists():
        FAILURE_OUTPUT.unlink()

    if PROGRESS_OUTPUT.exists():
        PROGRESS_OUTPUT.unlink()

    print("\n==============================================")
    print(" SEGMENTATION V2 COMPLETED")
    print("==============================================")
    print("Total input images:", len(df))
    print("Successfully segmented:", segmented_count)
    print("Rejected masks:", rejected_count)
    print("Processing failures:", processing_failed_count)
    print("Preview images:", sum(preview_counts.values()))

    if not feature_df.empty:
        print("\nBatch distribution:")
        print(feature_df.groupby("batch").size())
        print("\nCircle detection methods:")
        print(feature_df["circle_method"].value_counts())
        print("\nMask area statistics:")
        print(feature_df["mask_area_ratio"].describe())

    print("\nFeature CSV:", FEATURE_OUTPUT)
    print("Masks:", MASK_DIR)
    print("Overlays:", OVERLAY_DIR)
    print("Previews:", PREVIEW_DIR)
    if failed_rows:
        print("Failure report:", FAILURE_OUTPUT)
    print("\n==============================================")


if __name__ == "__main__":
    main()
