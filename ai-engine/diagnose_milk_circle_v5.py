"""
MilkOptic AI - V5 Hough Diagnostic (NOT final segmentation)
Purpose: explain why V4 found zero inner-circle candidates.
Creates diagnostic previews for up to 2 images per batch/concentration/zoom group.
Does not modify V1-V4, raw images, or model files.

Run:
    python diagnose_milk_circle_v5.py
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "analysis_output" / "camera_experiments" / "unified_all_batches.csv"
RAW_DIR = ROOT / "research_data" / "phone_photo" / "raw"
OUT_DIR = ROOT / "analysis_output" / "milk_segmentation_v5_diagnostics"
PREVIEW_DIR = OUT_DIR / "previews"

CROP_FRACTION = 0.42
WORK_SIZE = 640
PREVIEWS_PER_GROUP = 2

# Deliberately broad diagnostic sweep. These are exploratory candidates,
# not accepted masks and not model-training features.
PARAM2_VALUES = (30, 26, 22, 19, 16, 13, 10)
RADIUS_RANGES = ((35, 190), (45, 175), (55, 160))


def resolve_image_path(row):
    rel = str(row.get("relative_path", "")).replace("/", "\\").strip("\\")
    batch = str(row.get("batch", "")).strip()
    concentration = str(row.get("concentration", "")).strip()
    candidates = []

    if batch.upper() == "B1":
        candidates.append(RAW_DIR / concentration / Path(rel))
    else:
        candidates.append(RAW_DIR / Path(rel))

    name = str(row.get("file_name", "")).strip()
    if name:
        candidates.append(RAW_DIR / name)

    for p in candidates:
        if p.exists():
            return p

    if name:
        found = next(RAW_DIR.rglob(name), None)
        if found:
            return found
    return None


def make_crop(image):
    h, w = image.shape[:2]
    side = max(1, int(min(h, w) * CROP_FRACTION))
    x1 = max(0, w // 2 - side // 2)
    y1 = max(0, h // 2 - side // 2)
    crop = image[y1:y1 + side, x1:x1 + side]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)


def run_hough(gray_blur):
    h, w = gray_blur.shape
    candidates = []
    seen = set()

    for min_r, max_r in RADIUS_RANGES:
        for param2 in PARAM2_VALUES:
            circles = cv2.HoughCircles(
                gray_blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=55,
                param1=100,
                param2=param2,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is None:
                continue
            for x, y, r in np.round(circles[0]).astype(int):
                # Keep all detections in diagnostic image, but mark central ones separately.
                key = (int(x), int(y), int(r))
                if key in seen:
                    continue
                seen.add(key)
                offset = float(np.hypot(x - w / 2, y - h / 2) / (min(h, w) / 2))
                candidates.append({
                    "x": int(x), "y": int(y), "r": int(r),
                    "param2": int(param2),
                    "min_radius": int(min_r),
                    "max_radius": int(max_r),
                    "center_offset": offset,
                })

    # Sort near-center and smaller circles first; outer glass rim is usually larger.
    candidates.sort(key=lambda c: (c["center_offset"], c["r"]))
    return candidates


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    counts = {}
    rows = []
    image_errors = 0
    preview_count = 0

    for idx, row in df.iterrows():
        group = (
            str(row.get("batch", "UNKNOWN")),
            str(row.get("concentration", "UNKNOWN")),
            str(row.get("zoom", "UNKNOWN")),
        )
        if counts.get(group, 0) >= PREVIEWS_PER_GROUP:
            continue

        image_path = resolve_image_path(row)
        status = "ok"
        reason = ""

        try:
            if image_path is None:
                raise FileNotFoundError("Image path could not be resolved")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV could not decode image")

            crop = make_crop(image)
            if crop is None:
                raise ValueError("Central crop empty")

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 1.8)
            edges = cv2.Canny(blur, 25, 90)
            candidates = run_hough(blur)

            # Panel 1: crop
            p1 = crop.copy()

            # Panel 2: edge map
            p2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Panel 3: all Hough candidates, color-coded by radius
            p3 = crop.copy()
            central_candidates = [
                c for c in candidates
                if c["center_offset"] <= 0.22 and c["r"] <= 175
            ]
            for c in candidates:
                color = (0, 255, 0) if c in central_candidates else (0, 165, 255)
                cv2.circle(p3, (c["x"], c["y"]), c["r"], color, 2)
                cv2.circle(p3, (c["x"], c["y"]), 3, (0, 0, 255), -1)

            # Panel 4: best central/small candidate highlighted, or explicit no-candidate label.
            p4 = crop.copy()
            if central_candidates:
                best = central_candidates[0]
                cv2.circle(p4, (best["x"], best["y"]), best["r"], (0, 255, 0), 3)
                cv2.putText(
                    p4,
                    f"candidate r={best['r']} p2={best['param2']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    p4, "NO CENTRAL INNER CIRCLE CANDIDATE",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2
                )

            panels = [p1, p2, p3, p4]
            labels = ["CROP", "CANNY EDGES", "HOUGH CANDIDATES", "BEST CENTRAL"]
            for panel, label in zip(panels, labels):
                cv2.putText(
                    panel, label, (10, WORK_SIZE - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
                )

            stem = f"{idx:05d}_{group[0]}_{group[1]}_{group[2]}".replace("/", "_")
            out_file = PREVIEW_DIR / f"{stem}_diagnostic.jpg"
            cv2.imwrite(str(out_file), np.hstack(panels))

            counts[group] = counts.get(group, 0) + 1
            preview_count += 1

            rows.append({
                "source_row": int(idx),
                "batch": group[0],
                "concentration": group[1],
                "zoom": group[2],
                "image_path": str(image_path),
                "candidate_count": len(candidates),
                "central_candidate_count": len(central_candidates),
                "candidates": str(candidates[:15]),
                "preview_path": str(out_file),
                "status": status,
                "reason": reason,
            })

        except Exception as exc:
            image_errors += 1
            rows.append({
                "source_row": int(idx),
                "batch": group[0],
                "concentration": group[1],
                "zoom": group[2],
                "image_path": str(image_path) if image_path else "",
                "candidate_count": 0,
                "central_candidate_count": 0,
                "candidates": "",
                "preview_path": "",
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })

        if preview_count and preview_count % 10 == 0:
            print(f"Diagnostic previews created: {preview_count}")

    report = pd.DataFrame(rows)
    report.to_csv(OUT_DIR / "hough_diagnostic_report.csv", index=False)

    print("\nV5 DIAGNOSTICS COMPLETE")
    print(f"Diagnostic previews: {preview_count}")
    print(f"Image errors: {image_errors}")
    print(f"Report: {OUT_DIR / 'hough_diagnostic_report.csv'}")
    print(f"Previews: {PREVIEW_DIR}")
    print("These are candidate visualizations only; no segmentation mask is accepted.")


if __name__ == "__main__":
    main()
