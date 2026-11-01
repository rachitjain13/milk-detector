"""Experimental top-down glass/liquid candidate detector for MilkOptic AI.

This is a capture-quality helper, NOT a milk identity or adulteration test.
It detects a likely circular glass rim and returns a central liquid candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Camera setup used during the current experiment.
CAMERA_HEIGHT_CM = 20
ZOOM_PROFILES = {
    "0.5": {"min_radius_ratio": 0.18, "max_radius_ratio": 0.49},
    "1.0": {"min_radius_ratio": 0.18, "max_radius_ratio": 0.49},
}


def _fail(message: str, **extra):
    return {
        "valid": False,
        "glass_detected": False,
        "milk_like_candidate": False,
        "milk_presence_verified": False,
        "message": message,
        **extra,
    }


def detect_glass_and_milk(image_path: str, zoom: str = "1.0", save_preview: bool = True):
    if zoom not in ZOOM_PROFILES:
        return _fail("Unknown zoom profile. Use 0.5 or 1.0.")

    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return _fail("Image could not be opened.")

    h, w = image.shape[:2]
    if min(h, w) < 500:
        return _fail("Image resolution is too low. Capture at least 500 pixels on the shorter side.", image_width=w, image_height=h)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    profile = ZOOM_PROFILES[zoom]
    min_radius = max(30, int(min(h, w) * profile["min_radius_ratio"]))
    max_radius = int(min(h, w) * profile["max_radius_ratio"])

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(30, int(min(h, w) * 0.30)),
        param1=100,
        param2=35,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return _fail("Glass rim not found. Centre the complete glass and reduce reflections.", image_width=w, image_height=h, zoom_profile=zoom)

    candidates = np.round(circles[0]).astype(int)
    image_cx, image_cy = w / 2.0, h / 2.0
    ranked = []
    for cx, cy, radius in candidates:
        offset = np.hypot((cx - image_cx) / max(w, 1), (cy - image_cy) / max(h, 1))
        # Prefer a large, near-centre circular rim. Allow moderate framing error.
        if offset <= 0.28:
            score = (radius / min(h, w)) - (offset * 0.65)
            ranked.append((score, cx, cy, radius, offset))

    if not ranked:
        return _fail("A circular edge was found, but it is too far from the image centre. Reposition the glass.", image_width=w, image_height=h, zoom_profile=zoom)

    _, cx, cy, radius, offset = max(ranked, key=lambda item: item[0])

    # The inner liquid region is a conservative central disk. It intentionally
    # avoids the rim; it is a candidate ROI, not a verified milk segmentation.
    inner_radius = int(radius * 0.58)
    yy, xx = np.ogrid[:h, :w]
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inner_mask = distance <= inner_radius
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray_full[inner_mask]
    saturation = hsv[:, :, 1][inner_mask]

    if values.size == 0:
        return _fail("Could not extract the central glass region.", image_width=w, image_height=h, zoom_profile=zoom)

    brightness_mean = float(np.mean(values))
    saturation_mean = float(np.mean(saturation))
    bright_fraction = float(np.mean(values > 100))
    # Kept deliberately descriptive and non-authoritative.
    milk_like_candidate = brightness_mean > 100 and saturation_mean < 90 and bright_fraction > 0.80

    preview_path = None
    if save_preview:
        output_dir = path.parent / "calibration_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        preview = image.copy()
        cv2.circle(preview, (int(cx), int(cy)), int(radius), (0, 255, 0), 5)
        cv2.circle(preview, (int(cx), int(cy)), inner_radius, (0, 165, 255), 4)
        cv2.putText(preview, f"GLASS CANDIDATE - {zoom}x", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        preview_file = output_dir / f"glass_detection_{zoom.replace('.', '_')}.jpg"
        if cv2.imwrite(str(preview_file), preview):
            preview_path = str(preview_file)

    return {
        # Do not mark this as a validated capture until negative tests and
        # real-world calibration have been completed.
        "valid": False,
        "glass_detected": True,
        "glass_detection_status": "candidate_circle_found",
        "milk_like_candidate": bool(milk_like_candidate),
        "milk_presence_verified": False,
        "zoom_profile": zoom,
        "camera_height_cm": CAMERA_HEIGHT_CM,
        "image_width": int(w),
        "image_height": int(h),
        "glass_circle": {"center_x": int(cx), "center_y": int(cy), "radius": int(radius)},
        "centre_offset_fraction": round(float(offset), 4),
        "liquid_region": {
            "radius": int(inner_radius),
            "brightness_mean": round(brightness_mean, 2),
            "saturation_mean": round(saturation_mean, 2),
            "bright_pixel_fraction": round(bright_fraction, 4),
        },
        "preview_path": preview_path,
        "message": "Glass boundary is a visual candidate. Milk presence is not verified; experimental capture review is required.",
    }


def main():
    parser = argparse.ArgumentParser(description="MilkOptic experimental glass capture detector")
    parser.add_argument("image_path")
    parser.add_argument("--zoom", choices=["0.5", "1.0"], default="1.0")
    args = parser.parse_args()
    print(json.dumps(detect_glass_and_milk(args.image_path, args.zoom), indent=2))


if __name__ == "__main__":
    main()
