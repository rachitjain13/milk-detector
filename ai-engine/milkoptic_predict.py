#!/usr/bin/env python3
"""MilkOptic AI - ROI70 Ridge prediction using the saved V6-compatible bundle."""
from pathlib import Path
import argparse
import json
import sys

import cv2
import joblib
import numpy as np

MAX_IMAGE_SIDE = 1000
ROI_PERCENTAGE = 70

def resize_image(image, max_side=MAX_IMAGE_SIDE):
    h, w = image.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return image

def make_center_ellipse_mask(height, width, percentage):
    mask = np.zeros((height, width), dtype=np.uint8)
    diameter = max(8, int(round(min(height, width) * percentage / 100.0)))
    radius = max(4, diameter // 2)
    center = (width // 2, height // 2)
    cv2.ellipse(mask, center, (radius, radius), 0, 0, 360, 255, -1)
    return mask

def safe_stats(values, prefix):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {f"{prefix}_{key}": float("nan")
                for key in ("mean", "std", "median", "p10", "p90")}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }

def extract_roi_features(image_bgr, mask):
    selected = mask > 0
    pixels_bgr = image_bgr[selected]
    if pixels_bgr.size == 0:
        raise ValueError("ROI mask contains no pixels")

    pixels_rgb = cv2.cvtColor(
        pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB
    ).reshape(-1, 3)
    pixels_hsv = cv2.cvtColor(
        pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)

    features = {}
    for index, channel in enumerate(("r", "g", "b")):
        features.update(safe_stats(pixels_rgb[:, index], f"rgb_{channel}"))
    for index, channel in enumerate(("h", "s", "v")):
        features.update(safe_stats(pixels_hsv[:, index], f"hsv_{channel}"))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_values = gray[selected]
    features.update(safe_stats(gray_values, "gray"))

    hist = cv2.calcHist([gray], [0], mask, [16], [0, 256]).flatten()
    hist_sum = float(hist.sum())
    if hist_sum > 0:
        hist /= hist_sum
    for index, value in enumerate(hist):
        features[f"gray_hist_{index:02d}"] = float(value)

    edges = cv2.Canny(gray, 60, 140)
    features["edge_fraction"] = float(np.mean(edges[selected] > 0))
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    features["laplacian_variance"] = float(np.var(laplacian[selected]))
    features["roi_pixel_count"] = int(np.count_nonzero(mask))
    features["roi_fraction_image"] = float(np.count_nonzero(mask) / mask.size)
    return features


def main():
    parser = argparse.ArgumentParser(
        description="Estimate milk-water dilution from an image."
    )
    parser.add_argument("image", help="Path to the captured milk image")

    BASE_DIR = Path(__file__).resolve().parent

    parser.add_argument(
        "--model-dir",
        default=str(BASE_DIR / "models" / "roi70_ridge"),
        help="Folder containing milkoptic_roi70_ridge.joblib"
    )

    args = parser.parse_args()


    model_path = Path(args.model_dir) / "milkoptic_roi70_ridge.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model bundle not found: {model_path}")

    # The saved .joblib is a dictionary bundle; the fitted sklearn Pipeline is bundle["pipeline"].
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected a dictionary model bundle, got {type(bundle).__name__}")
    if "pipeline" not in bundle or not hasattr(bundle["pipeline"], "predict"):
        raise ValueError("Model bundle does not contain a fitted prediction pipeline.")
    expected_features = bundle.get("feature_names")
    if not isinstance(expected_features, list) or not expected_features:
        raise ValueError("Model bundle does not contain feature_names.")

    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Match V6 training preprocessing: resize first, then 70% centered ellipse, then features.
    image = resize_image(image)
    height, width = image.shape[:2]
    mask = make_center_ellipse_mask(height, width, ROI_PERCENTAGE)
    features = extract_roi_features(image, mask)

    missing = [name for name in expected_features if name not in features]
    extra = [name for name in features if name not in expected_features]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    x = np.asarray([[features[name] for name in expected_features]], dtype=np.float64)
    pipeline = bundle["pipeline"]
    raw_prediction = float(np.asarray(pipeline.predict(x)).reshape(-1)[0])
    prediction = float(np.clip(raw_prediction, 0.0, 100.0))

    result = {
        "success": True,
        "prediction_type": "estimated_water_dilution_percent",
        "estimated_water_dilution_percent": round(prediction, 2),
        "raw_model_output": round(raw_prediction, 4),
        "roi_percentage": int(bundle.get("roi_size", ROI_PERCENTAGE)),
        "image_size_after_resize": {"width": width, "height": height},
        "feature_count": len(expected_features),
        "missing_features": missing,
        "unexpected_extracted_features": extra,
        "model_name": bundle.get("model_name", "ROI70 Ridge"),
        "feature_set": bundle.get("feature_set"),
        "training_batch": bundle.get("training_batch"),
        "validation_note": (
            "Experimental estimate only. Internal B1 grouped validation; "
            "not independently validated for real-world/lab accuracy."
        ),
        "scope": "Estimates water dilution only; does not identify detergent, soybean, or other adulterants."
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2), file=sys.stderr)
        sys.exit(1)
