#!/usr/bin/env python
"""
MilkOptic AI - phone image feature extractor
Extracts descriptive features from ordinary milk-in-glass photos.
This does NOT diagnose adulteration or estimate percentages.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def safe_stats(a, prefix):
    a = np.asarray(a, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_std": float(np.std(a)),
        f"{prefix}_min": float(np.min(a)),
        f"{prefix}_max": float(np.max(a)),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_p10": float(np.percentile(a, 10)),
        f"{prefix}_p90": float(np.percentile(a, 90)),
    }

def extract(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    # Consistent maximum size for feature extraction; preserve original file.
    h0, w0 = bgr.shape[:2]
    scale = min(1.0, 640.0 / max(h0, w0))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = {
        "file_name": path.name,
        "image_width_original": int(w0),
        "image_height_original": int(h0),
        "image_mean_brightness_gray": float(np.mean(gray)),
        "image_brightness_std_gray": float(np.std(gray)),
        "image_blur_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "image_edge_density": float(np.mean(cv2.Canny(gray, 80, 160) > 0)),
    }

    for i, channel in enumerate(("r", "g", "b")):
        out.update(safe_stats(rgb[:, :, i], f"rgb_{channel}"))
    for i, channel in enumerate(("h", "s", "v")):
        out.update(safe_stats(hsv[:, :, i], f"hsv_{channel}"))

    # Normalized grayscale histogram: 16 bins
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist = hist / max(hist.sum(), 1)
    for i, value in enumerate(hist):
        out[f"gray_hist_bin_{i:02d}"] = float(value)

    # Texture features from quantized grayscale. These describe the whole image;
    # they do not automatically isolate the milk region from glass/background.
    gray_small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    quant = (gray_small // 16).astype(np.uint8)
    glcm = graycomatrix(
        quant, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=16, symmetric=True, normed=True
    )
    for prop in ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"):
        vals = graycoprops(glcm, prop)
        out[f"glcm_{prop}_mean"] = float(np.mean(vals))
        out[f"glcm_{prop}_std"] = float(np.std(vals))

    lbp = local_binary_pattern(gray_small, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10), density=True)
    for i, value in enumerate(lbp_hist):
        out[f"lbp_bin_{i:02d}"] = float(value)

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing original phone images")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)
    if not input_dir.exists():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No supported images found in: {input_dir}")

    rows, errors = [], []
    for p in paths:
        try:
            row = extract(p)
            if row is not None:
                row["relative_path"] = str(p.relative_to(input_dir))
                rows.append(row)
            else:
                errors.append(str(p))
        except Exception as e:
            errors.append(f"{p}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Images processed: {len(rows)}")
    print(f"Failed images: {len(errors)}")
    print(f"Feature columns (excluding file/path): {max(0, len(rows[0])-2) if rows else 0}")
    print(f"CSV saved: {output_path}")
    if errors:
        print("First errors:")
        for e in errors[:10]:
            print(" -", e)

if __name__ == "__main__":
    main()
