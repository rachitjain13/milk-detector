"""
MilkOptic AI - Milk Surface Segmentation V4
Detects the INNER milk-surface circle inside the glass.
Does not overwrite raw data or V1/V2/V3 outputs.

Run from ai-engine:
    python segment_milk_region_v4.py

Input:
    analysis_output/camera_experiments/unified_all_batches.csv
    research_data/phone_photo/raw/

Output:
    analysis_output/milk_segmentation_v4/
"""

from pathlib import Path
import math
import warnings

import cv2
import numpy as np
import pandas as pd

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
except ImportError as exc:
    raise SystemExit(
        "Missing scikit-image. Install it with: pip install scikit-image"
    ) from exc


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "analysis_output" / "camera_experiments" / "unified_all_batches.csv"
RAW_DIR = ROOT / "research_data" / "phone_photo" / "raw"
OUT_DIR = ROOT / "analysis_output" / "milk_segmentation_v4"

MASK_DIR = OUT_DIR / "masks"
OVERLAY_DIR = OUT_DIR / "overlays"
PREVIEW_DIR = OUT_DIR / "previews"

# Keep the central crop used in earlier experiments, then resize for stable RAM use.
CROP_FRACTION = 0.42
WORK_SIZE = 640
PREVIEW_SIZE = 512
PREVIEWS_PER_GROUP = 2

# The milk surface is the smaller inner circle, not the larger glass rim.
MIN_RADIUS_FRAC = 0.105
MAX_RADIUS_FRAC = 0.265
MAX_CENTER_OFFSET_FRAC = 0.18
MASK_INSET = 0.96

# Conservative acceptance thresholds. Rejected images are not assigned fake masks.
MIN_EDGE_SUPPORT = 0.105
MIN_CONTRAST = 2.0
MIN_CONFIDENCE = 0.19


def resolve_image_path(row):
    rel = str(row.get("relative_path", "")).replace("/", "\\").strip("\\")
    batch = str(row.get("batch", "")).strip()
    concentration = str(row.get("concentration", "")).strip()

    candidates = []
    if batch.upper() == "B1":
        candidates.append(RAW_DIR / concentration / Path(rel))
    else:
        candidates.append(RAW_DIR / Path(rel))

    file_name = str(row.get("file_name", "")).strip()
    if file_name:
        candidates.append(RAW_DIR / file_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Last-resort exact filename search; only used when manifest path layout differs.
    if file_name:
        matches = list(RAW_DIR.rglob(file_name))
        if matches:
            return matches[0]
    return None


def center_crop_square(image, fraction=CROP_FRACTION):
    h, w = image.shape[:2]
    side = max(1, int(min(h, w) * fraction))
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)


def sample_circle_values(array, cx, cy, radius, count=360):
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    xs = np.clip(np.round(cx + radius * np.cos(angles)).astype(int), 0, array.shape[1] - 1)
    ys = np.clip(np.round(cy + radius * np.sin(angles)).astype(int), 0, array.shape[0] - 1)
    return array[ys, xs]


def evaluate_circle(gray, edge_map, grad_mag, cx, cy, radius):
    h, w = gray.shape
    center_dist = math.hypot(cx - w / 2, cy - h / 2) / (min(h, w) / 2)

    circumference_edges = sample_circle_values(edge_map, cx, cy, radius)
    # Permit a 5-pixel tolerance around the predicted boundary.
    support_values = []
    for offset in (-5, -3, -1, 0, 1, 3, 5):
        support_values.append(
            sample_circle_values(edge_map, cx, cy, max(2, radius + offset))
        )
    support = float(np.mean(np.max(np.stack(support_values), axis=0) > 0))

    boundary_gradient = sample_circle_values(grad_mag, cx, cy, radius)
    gradient_score = float(np.percentile(boundary_gradient, 65))
    gradient_norm = min(1.0, gradient_score / 55.0)

    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inner = dist2 <= (radius * 0.78) ** 2
    ring = (dist2 >= (radius * 1.08) ** 2) & (dist2 <= (radius * 1.28) ** 2)
    if inner.sum() < 100 or ring.sum() < 100:
        return None

    inner_mean = float(np.mean(gray[inner]))
    ring_mean = float(np.mean(gray[ring]))
    contrast = abs(inner_mean - ring_mean)
    contrast_score = min(1.0, contrast / 28.0)

    center_score = max(0.0, 1.0 - center_dist / MAX_CENTER_OFFSET_FRAC)
    confidence = (
        0.45 * support
        + 0.25 * gradient_norm
        + 0.20 * contrast_score
        + 0.10 * center_score
    )

    return {
        "cx": float(cx),
        "cy": float(cy),
        "radius": float(radius),
        "edge_support": support,
        "gradient_score": gradient_score,
        "contrast": contrast,
        "center_offset_fraction": center_dist,
        "confidence": float(confidence),
        "inner_mean_gray": inner_mean,
        "surrounding_mean_gray": ring_mean,
    }


def detect_inner_milk_circle(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 1.8)
    edges = cv2.Canny(gray_blur, 35, 105)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    min_dim = min(gray.shape)
    min_radius = int(min_dim * MIN_RADIUS_FRAC)
    max_radius = int(min_dim * MAX_RADIUS_FRAC)

    candidates = []
    # Multiple Hough thresholds improve recall; candidate scoring controls acceptance.
    for param2 in (30, 26, 22, 19):
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min_dim * 0.12),
            param1=100,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            continue
        for c in np.round(circles[0]).astype(int):
            x, y, r = map(int, c)
            offset = math.hypot(x - min_dim / 2, y - min_dim / 2) / (min_dim / 2)
            if offset > MAX_CENTER_OFFSET_FRAC:
                continue
            result = evaluate_circle(gray, edges, grad_mag, x, y, r)
            if result is not None:
                candidates.append(result)

    if not candidates:
        return None, gray, edges, "rejected_no_inner_circle_candidate"

    # Prefer strong, central inner boundaries; the radius cap excludes the outer glass rim.
    candidates.sort(
        key=lambda c: (
            c["confidence"],
            c["edge_support"],
            c["contrast"],
            -c["center_offset_fraction"],
        ),
        reverse=True,
    )
    best = candidates[0]

    if best["edge_support"] < MIN_EDGE_SUPPORT:
        return None, gray, edges, "rejected_weak_boundary_support"
    if best["contrast"] < MIN_CONTRAST:
        return None, gray, edges, "rejected_low_inner_outer_contrast"
    if best["confidence"] < MIN_CONFIDENCE:
        return None, gray, edges, "rejected_low_confidence"

    return best, gray, edges, "segmented_inner_milk_circle"


def extract_masked_features(crop, mask):
    pixels = crop[mask > 0]
    if len(pixels) < 100:
        return None

    features = {}
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    for name, channel in zip(("r", "g", "b"), cv2.split(rgb)):
        vals = channel[mask > 0].astype(np.float32)
        features[f"{name}_mean"] = float(np.mean(vals))
        features[f"{name}_std"] = float(np.std(vals))
        features[f"{name}_median"] = float(np.median(vals))
        features[f"{name}_p10"] = float(np.percentile(vals, 10))
        features[f"{name}_p90"] = float(np.percentile(vals, 90))

    for name, channel in zip(("h", "s", "v"), cv2.split(hsv)):
        vals = channel[mask > 0].astype(np.float32)
        features[f"{name}_mean"] = float(np.mean(vals))
        features[f"{name}_std"] = float(np.std(vals))
        features[f"{name}_median"] = float(np.median(vals))
        features[f"{name}_p10"] = float(np.percentile(vals, 10))
        features[f"{name}_p90"] = float(np.percentile(vals, 90))

    gray_vals = gray[mask > 0].astype(np.float32)
    features["gray_mean"] = float(np.mean(gray_vals))
    features["gray_std"] = float(np.std(gray_vals))
    features["gray_median"] = float(np.median(gray_vals))
    features["gray_p10"] = float(np.percentile(gray_vals, 10))
    features["gray_p90"] = float(np.percentile(gray_vals, 90))

    hist = cv2.calcHist([gray], [0], mask, [16], [0, 256]).flatten()
    hist = hist / max(float(hist.sum()), 1.0)
    for i, value in enumerate(hist):
        features[f"gray_hist_{i:02d}"] = float(value)

    # Texture is computed only from the milk mask bounding region. Outside-mask pixels
    # are filled with the milk-pixel median to avoid a black background edge.
    filled = gray.copy()
    filled[mask == 0] = int(np.median(gray_vals))
    small = cv2.resize(filled, (256, 256), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    glcm_img = (small // 4).astype(np.uint8)
    glcm = graycomatrix(
        glcm_img,
        distances=[1, 3],
        angles=[0, np.pi / 4],
        levels=64,
        symmetric=True,
        normed=True,
    )
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        vals = graycoprops(glcm, prop)
        features[f"glcm_{prop}_mean"] = float(np.mean(vals))
        features[f"glcm_{prop}_std"] = float(np.std(vals))

    lbp = local_binary_pattern(small, P=8, R=1, method="uniform")
    lbp_mask = small_mask > 0
    lbp_hist, _ = np.histogram(lbp[lbp_mask], bins=np.arange(0, 12), density=True)
    for i, value in enumerate(lbp_hist):
        features[f"lbp_{i:02d}"] = float(value)

    features["milk_mask_pixel_count"] = int(np.count_nonzero(mask))
    features["milk_mask_area_ratio"] = float(np.count_nonzero(mask) / mask.size)
    return features


def make_preview(crop, edges, mask, circle, out_path):
    base = cv2.resize(crop, (PREVIEW_SIZE, PREVIEW_SIZE))
    edge_panel = cv2.cvtColor(
        cv2.resize(edges, (PREVIEW_SIZE, PREVIEW_SIZE)), cv2.COLOR_GRAY2BGR
    )
    mask_panel = cv2.cvtColor(
        cv2.resize(mask, (PREVIEW_SIZE, PREVIEW_SIZE)), cv2.COLOR_GRAY2BGR
    )
    overlay = base.copy()
    green = np.zeros_like(base)
    green[:, :, 1] = 255
    m = mask > 0
    overlay[m] = cv2.addWeighted(base[m], 0.45, green[m], 0.55, 0)
    if circle:
        scale = PREVIEW_SIZE / WORK_SIZE
        center = (int(circle["cx"] * scale), int(circle["cy"] * scale))
        radius = int(circle["radius"] * scale)
        cv2.circle(overlay, center, radius, (0, 0, 255), 2)
    panels = [base, edge_panel, mask_panel, overlay]
    labeled = []
    for panel, label in zip(panels, ("CROP", "EDGES", "MILK MASK", "OVERLAY")):
        cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        labeled.append(panel)
    cv2.imwrite(str(out_path), np.hstack(labeled))


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    required = {"batch", "concentration", "relative_path"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Input CSV missing required columns: {sorted(missing)}")

    feature_rows = []
    audit_rows = []
    preview_counts = {}
    failure_count = 0
    segmented_count = 0
    rejected_count = 0

    for idx, row in df.iterrows():
        batch = str(row.get("batch", "UNKNOWN"))
        concentration = str(row.get("concentration", "UNKNOWN"))
        zoom = str(row.get("zoom", "UNKNOWN"))
        key = (batch, concentration, zoom)

        image_path = resolve_image_path(row)
        audit = {
            "source_row": int(idx),
            "batch": batch,
            "concentration": concentration,
            "zoom": zoom,
            "relative_path": str(row.get("relative_path", "")),
            "image_path": str(image_path) if image_path else "",
            "status": "",
            "reason": "",
        }

        try:
            if image_path is None:
                raise FileNotFoundError("Image path could not be resolved")

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV could not decode image")

            crop = center_crop_square(image)
            if crop is None:
                raise ValueError("Central crop is empty")

            circle, gray, edges, status = detect_inner_milk_circle(crop)
            audit["status"] = status

            if circle is None:
                audit["reason"] = status
                rejected_count += 1
                audit_rows.append(audit)
                continue

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(
                mask,
                (int(round(circle["cx"])), int(round(circle["cy"]))),
                max(1, int(round(circle["radius"] * MASK_INSET))),
                255,
                -1,
            )

            features = extract_masked_features(crop, mask)
            if features is None:
                audit["status"] = "rejected_insufficient_mask_pixels"
                audit["reason"] = "Too few pixels inside candidate milk circle"
                rejected_count += 1
                audit_rows.append(audit)
                continue

            stem = f"{idx:05d}_{batch}_{concentration}_{zoom}".replace("/", "_")
            cv2.imwrite(str(MASK_DIR / f"{stem}_mask.png"), mask)

            overlay = crop.copy()
            green = np.zeros_like(crop)
            green[:, :, 1] = 255
            selected = mask > 0
            overlay[selected] = cv2.addWeighted(
                crop[selected], 0.45, green[selected], 0.55, 0
            )
            cv2.circle(
                overlay,
                (int(round(circle["cx"])), int(round(circle["cy"]))),
                int(round(circle["radius"])),
                (0, 0, 255),
                2,
            )
            cv2.imwrite(str(OVERLAY_DIR / f"{stem}_overlay.jpg"), overlay)

            if preview_counts.get(key, 0) < PREVIEWS_PER_GROUP:
                make_preview(
                    crop, edges, mask, circle, PREVIEW_DIR / f"{stem}_preview.jpg"
                )
                preview_counts[key] = preview_counts.get(key, 0) + 1

            feature_row = row.to_dict()
            feature_row.update(features)
            feature_row.update({
                "segmentation_method": "inner_circle_hough_scored",
                "circle_center_x": circle["cx"],
                "circle_center_y": circle["cy"],
                "circle_radius": circle["radius"],
                "boundary_edge_support": circle["edge_support"],
                "boundary_gradient_score": circle["gradient_score"],
                "inner_outer_gray_contrast": circle["contrast"],
                "boundary_confidence": circle["confidence"],
                "center_offset_fraction": circle["center_offset_fraction"],
            })
            feature_rows.append(feature_row)

            audit.update({
                "status": "segmented_inner_milk_circle",
                "reason": "",
                "circle_center_x": circle["cx"],
                "circle_center_y": circle["cy"],
                "circle_radius": circle["radius"],
                "mask_area_ratio": features["milk_mask_area_ratio"],
                "boundary_edge_support": circle["edge_support"],
                "inner_outer_gray_contrast": circle["contrast"],
                "boundary_confidence": circle["confidence"],
            })
            segmented_count += 1

        except Exception as exc:
            failure_count += 1
            audit["status"] = "failed_processing_error"
            audit["reason"] = f"{type(exc).__name__}: {exc}"

        audit_rows.append(audit)

        if (idx + 1) % 100 == 0 or idx + 1 == len(df):
            pd.DataFrame(feature_rows).to_csv(
                OUT_DIR / "segmented_milk_features.csv", index=False
            )
            pd.DataFrame(audit_rows).to_csv(
                OUT_DIR / "segmentation_audit.csv", index=False
            )
            print(
                f"{idx + 1}/{len(df)} | segmented={segmented_count} "
                f"rejected={rejected_count} failed={failure_count}"
            )

    feature_path = OUT_DIR / "segmented_milk_features.csv"
    audit_path = OUT_DIR / "segmentation_audit.csv"
    if not feature_path.exists():
        pd.DataFrame(feature_rows).to_csv(feature_path, index=False)
    if not audit_path.exists():
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    print("\nV4 COMPLETE")
    print(f"Input: {len(df)} | Segmented: {segmented_count} | "
          f"Rejected: {rejected_count} | Failed: {failure_count}")
    print(f"Features: {feature_path}")
    print(f"Audit: {audit_path}")
    print(f"Previews: {PREVIEW_DIR}")
    print("IMPORTANT: Inspect V4 previews before using these features for model training.")


if __name__ == "__main__":
    main()
