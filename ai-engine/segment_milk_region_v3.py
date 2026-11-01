"""MilkOptic AI segmentation V3: adaptive inner-rim detection.

Run from ai-engine:
    python segment_milk_region_v3.py

Reads:
    analysis_output/camera_experiments/unified_all_batches.csv
    research_data/phone_photo/raw/

Writes only to:
    analysis_output/milk_segmentation_v3/

V3 deliberately rejects uncertain rim detections instead of silently using a
fixed circle. Inspect previews/audit before using the features for modelling.
"""
from pathlib import Path
import math
import re

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "research_data" / "phone_photo" / "raw"
INPUT = ROOT / "analysis_output" / "camera_experiments" / "unified_all_batches.csv"
OUT = ROOT / "analysis_output" / "milk_segmentation_v3"
MASKS, PREVIEWS = OUT / "masks", OUT / "previews"

MAX_SIDE = 720
MIN_RADIUS_FRAC = 0.25
MAX_RADIUS_FRAC = 0.49
MAX_CENTER_OFFSET_FRAC = 0.22
RIM_INSET_FRAC = 0.035
MIN_EDGE_SUPPORT = 0.18
PREVIEWS_PER_GROUP = 2


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def resolve_path(row):
    rel = Path(str(row["relative_path"]).replace("\\", "/"))
    if str(row["batch"]).upper() == "B1":
        candidate = RAW / str(row["concentration"]) / rel
    else:
        candidate = RAW / rel
    if candidate.exists():
        return candidate
    matches = list(RAW.rglob(rel.name))
    preferred = [p for p in matches if str(row["concentration"]).lower() in str(p).lower()]
    if len(preferred) == 1:
        return preferred[0]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not uniquely resolve {candidate}")


def prep_image(im):
    h, w = im.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1:
        im = cv2.resize(im, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return im, scale


def edges_for(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    med = float(np.median(gray))
    lo = int(max(20, 0.55 * med))
    hi = int(min(220, max(lo + 35, 1.35 * med)))
    return cv2.Canny(gray, lo, hi)


def circumference_support(edge, cx, cy, radius):
    h, w = edge.shape
    a = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    hits_best = np.zeros(a.size, dtype=np.uint8)
    for off in (-3, -1, 0, 1, 3):
        xs = np.rint(cx + (radius + off) * np.cos(a)).astype(int)
        ys = np.rint(cy + (radius + off) * np.sin(a)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        hits = np.zeros(a.size, dtype=np.uint8)
        hits[valid] = (edge[ys[valid], xs[valid]] > 0).astype(np.uint8)
        hits_best = np.maximum(hits_best, hits)
    return float(hits_best.mean())


def find_rim(im):
    h, w = im.shape[:2]
    md = min(h, w)
    edge = edges_for(im)
    gray = cv2.medianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 5)
    candidates = []
    for dp, p1, p2 in [(1.2, 70, 24), (1.2, 55, 20), (1.35, 65, 18), (1.5, 60, 16)]:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=max(30, int(0.25 * md)),
            param1=p1, param2=p2, minRadius=int(max(20, MIN_RADIUS_FRAC * md)),
            maxRadius=int(MAX_RADIUS_FRAC * md)
        )
        if circles is not None:
            candidates.extend(np.round(circles[0]).astype(int).tolist())

    unique = []
    for x, y, r in candidates:
        if not any(abs(x-a) < 8 and abs(y-b) < 8 and abs(r-c) < 10 for a, b, c in unique):
            unique.append((x, y, r))

    cx0, cy0 = w / 2, h / 2
    maxoff = MAX_CENTER_OFFSET_FRAC * md
    ranked = []
    for x, y, r in unique:
        off = math.hypot(x-cx0, y-cy0)
        if off > maxoff:
            continue
        support = circumference_support(edge, x, y, r)
        center_score = 1 - min(1, off / maxoff)
        score = 0.88 * support + 0.12 * center_score
        ranked.append((score, support, x, y, r))
    if not ranked:
        return None, edge
    ranked.sort(reverse=True)
    score, support, x, y, r = ranked[0]
    if support < MIN_EDGE_SUPPORT:
        return None, edge
    return {"cx": int(x), "cy": int(y), "r": int(r),
            "support": float(support), "score": float(score)}, edge


def make_mask(shape, circle):
    h, w = shape
    inset = max(3, round(circle["r"] * RIM_INSET_FRAC))
    radius = max(1, circle["r"] - inset)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (circle["cx"], circle["cy"]), radius, 255, -1)
    return mask, inset / max(circle["r"], 1)


def extract_features(im, mask):
    valid = mask > 0
    if not valid.any():
        raise ValueError("Empty mask")
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    out = {}
    for name, channel in zip(("R", "G", "B"), cv2.split(rgb)):
        v = channel[valid].astype(np.float32)
        for stat, val in [("mean", v.mean()), ("std", v.std()),
                          ("median", np.median(v)), ("p10", np.percentile(v, 10)),
                          ("p90", np.percentile(v, 90))]:
            out[f"{name}_{stat}"] = float(val)
    for name, channel in zip(("H", "S", "V"), cv2.split(hsv)):
        v = channel[valid].astype(np.float32)
        out[f"{name}_mean"] = float(v.mean())
        out[f"{name}_std"] = float(v.std())
        out[f"{name}_median"] = float(np.median(v))
    gv = gray[valid].astype(np.float32)
    for stat, val in [("mean", gv.mean()), ("std", gv.std()), ("median", np.median(gv)),
                      ("p10", np.percentile(gv, 10)), ("p90", np.percentile(gv, 90))]:
        out[f"gray_{stat}"] = float(val)
    hist = cv2.calcHist([gray], [0], mask, [16], [0, 256]).flatten()
    hist /= max(float(hist.sum()), 1.0)
    for i, value in enumerate(hist):
        out[f"gray_hist_{i:02d}"] = float(value)
    _, s, v = cv2.split(hsv)
    glare = valid & (v >= 245) & (s <= 45)
    out["glare_pixel_fraction"] = float(glare.sum() / valid.sum())
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    out["laplacian_variance_masked"] = float(np.var(lap[valid]))
    out["mask_pixel_count"] = int(valid.sum())
    out["mask_area_ratio"] = float(valid.sum() / mask.size)
    return out


def preview_panel(im, edge, mask):
    tint = np.zeros_like(im)
    tint[:] = (0, 220, 40)
    overlay = im.copy()
    sel = mask > 0
    overlay[sel] = cv2.addWeighted(im[sel], 0.48, tint[sel], 0.52, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return np.hstack([im, cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay])


def main():
    MASKS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT}")
    df = pd.read_csv(INPUT)
    required = {"batch", "concentration", "relative_path"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")

    features, audit, preview_counts = [], [], {}
    nseg = nrej = nfail = 0
    for i, row in df.iterrows():
        batch, conc, zoom = str(row["batch"]), str(row["concentration"]), str(row.get("zoom", "NA"))
        rel = str(row["relative_path"])
        record = {"row_index": i, "batch": batch, "concentration": conc,
                  "zoom": zoom, "relative_path": rel, "status": "",
                  "rim_edge_support": np.nan, "rim_score": np.nan,
                  "center_x": np.nan, "center_y": np.nan, "radius": np.nan,
                  "mask_area_ratio": np.nan, "error": ""}
        try:
            path = resolve_path(row)
            im = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError("OpenCV could not read image")
            im, scale = prep_image(im)
            circle, edge = find_rim(im)
            if circle is None:
                record["status"] = "rejected_rim_not_confident"
                record["error"] = "No sufficiently supported inner-rim circle"
                nrej += 1
            else:
                mask, inset = make_mask(im.shape[:2], circle)
                area = float(np.count_nonzero(mask) / mask.size)
                record.update({"rim_edge_support": circle["support"], "rim_score": circle["score"],
                               "center_x": circle["cx"], "center_y": circle["cy"],
                               "radius": circle["r"], "mask_area_ratio": area})
                if not 0.15 <= area <= 0.75:
                    record["status"] = "rejected_mask_area"
                    nrej += 1
                else:
                    feats = extract_features(im, mask)
                    meta = row.to_dict()
                    meta.update({"image_path": str(path), "segmentation_version": "V3",
                                 "segmentation_method": "hough_inner_rim",
                                 "rim_edge_support": circle["support"], "rim_score": circle["score"],
                                 "rim_center_x": circle["cx"], "rim_center_y": circle["cy"],
                                 "rim_radius": circle["r"], "rim_inset_fraction": inset,
                                 "processing_scale": scale})
                    meta.update(feats)
                    features.append(meta)
                    record["status"] = "segmented"
                    nseg += 1
                    stem = safe(Path(rel).stem)
                    uid = f"{i:05d}_{safe(batch)}_{safe(conc)}_{safe(zoom)}_{stem}"
                    cv2.imwrite(str(MASKS / f"{uid}_mask.png"), mask)
                    group = f"{batch}_{conc}_{zoom}"
                    if preview_counts.get(group, 0) < PREVIEWS_PER_GROUP:
                        panel = preview_panel(im, edge, mask)
                        cv2.imwrite(str(PREVIEWS / f"{uid}_preview.jpg"), panel,
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                        preview_counts[group] = preview_counts.get(group, 0) + 1
        except Exception as exc:
            record["status"] = "failed_processing"
            record["error"] = f"{type(exc).__name__}: {exc}"
            nfail += 1
            if nfail <= 20:
                print(f"[ERROR] row {i}: {record['error']}")
        audit.append(record)

        if (i + 1) % 100 == 0 or i + 1 == len(df):
            pd.DataFrame(features).to_csv(OUT / "segmented_milk_features.csv", index=False)
            pd.DataFrame(audit).to_csv(OUT / "segmentation_audit.csv", index=False)
            print(f"{i+1}/{len(df)} | segmented={nseg} rejected={nrej} failed={nfail}")

    pd.DataFrame(features).to_csv(OUT / "segmented_milk_features.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "segmentation_audit.csv", index=False)
    print("\nV3 COMPLETE")
    print(f"Input: {len(df)} | Segmented: {nseg} | Rejected: {nrej} | Failed: {nfail}")
    print(f"Features: {OUT / 'segmented_milk_features.csv'}")
    print(f"Audit: {OUT / 'segmentation_audit.csv'}")
    print(f"Previews: {PREVIEWS}")
    print("Do not train on these features until the previews have been inspected.")


if __name__ == "__main__":
    main()
