"""
MilkOptic AI - V6 Controlled Central ROI Experiment
===================================================

Compares three fixed central elliptical regions (60%, 70%, 80% of the
shorter image dimension) against the existing full-image feature baseline.

IMPORTANT:
- This is a controlled ROI experiment, NOT a true milk-boundary detector.
- It does not prove that selected pixels contain only milk.
- Original images, previous feature files, and models are not modified.
- B1/B2 are the only batches used; no B3 collection is needed.

Run from:
  D:\\avishkar\\MilkOptic-AI\\ai-engine

Command:
  python v6_controlled_roi_experiment.py

Input:
  analysis_output/camera_experiments/unified_all_batches.csv

Outputs:
  analysis_output/milk_segmentation_v6_roi/
"""

from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "analysis_output" / "camera_experiments" / "unified_all_batches.csv"
RAW_ROOT = ROOT / "research_data" / "phone_photo" / "raw"
OUT_ROOT = ROOT / "analysis_output" / "milk_segmentation_v6_roi"
PREVIEW_DIR = OUT_ROOT / "previews"
MODEL_DIR = OUT_ROOT / "model_comparison"

ROI_SIZES = (60, 70, 80)
MAX_IMAGE_SIDE = 1000
PREVIEWS_PER_GROUP_PER_SIZE = 2
RANDOM_STATE = 42


def resolve_image_path(relative_path, batch=None, concentration=None):
    """
    Resolve B1 and B2 paths.

    B1 manifest paths are relative to the concentration folder, e.g.
      0.5\\jpg\\image.jpg
    so B1 needs:
      raw/<concentration>/<relative_path>

    B2 paths are relative to raw, e.g.
      M00_B2\\0.5\\image.jpg
    so B2 needs:
      raw/<relative_path>
    """
    raw_text = str(relative_path).strip()
    if not raw_text or raw_text.lower() == "nan":
        return None

    p = Path(raw_text)
    if p.is_absolute() and p.exists():
        return p

    batch_text = str(batch).strip().upper()

    if batch_text == "B1" and concentration is not None:
        candidate = RAW_ROOT / str(concentration).strip() / p
        if candidate.exists():
            return candidate

    candidate = RAW_ROOT / p
    if candidate.exists():
        return candidate

    candidate = ROOT / p
    if candidate.exists():
        return candidate

    return None


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
    """
    Centered ellipse with diameter percentage of the shorter image dimension.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    diameter = max(8, int(round(min(height, width) * percentage / 100.0)))
    radius = max(4, diameter // 2)
    center = (width // 2, height // 2)
    cv2.ellipse(mask, center, (radius, radius), 0, 0, 360, 255, -1)
    return mask


def safe_stats(values, prefix):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            f"{prefix}_{key}": np.nan
            for key in ("mean", "std", "median", "p10", "p90")
        }

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


def save_preview(image, mask, percentage, output_path):
    overlay = image.copy()
    tint = np.zeros_like(image)
    tint[:] = (40, 190, 70)

    selected = mask > 0
    overlay[selected] = cv2.addWeighted(
        image[selected], 0.62, tint[selected], 0.38, 0
    )

    radius = max(4, int(round(min(image.shape[:2]) * percentage / 200.0)))
    cv2.ellipse(
        overlay,
        (image.shape[1] // 2, image.shape[0] // 2),
        (radius, radius),
        0, 0, 360, (0, 0, 255), 3
    )

    cv2.putText(
        overlay,
        f"V6 central ellipse {percentage}%",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise IOError(f"Could not save preview: {output_path}")


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def get_feature_columns(data, metadata_columns):
    return [
        column
        for column in data.columns
        if column not in metadata_columns
        and pd.api.types.is_numeric_dtype(data[column])
    ]


def evaluate_logo(data, feature_columns, experiment_name):
    data = data.reset_index(drop=True).copy()

    X = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = data["dilution_percent"].astype(float).to_numpy()
    groups = data["concentration"].astype(str).to_numpy()

    logo = LeaveOneGroupOut()
    predictions = np.full(len(data), np.nan, dtype=float)
    fold_rows = []

    for fold_number, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups), start=1
    ):
        model = build_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        fold_prediction = model.predict(X.iloc[test_idx])
        predictions[test_idx] = fold_prediction

        fold_rows.append({
            "experiment": experiment_name,
            "fold": fold_number,
            "held_out_concentration": str(groups[test_idx][0]),
            "n_test": int(len(test_idx)),
            "mae": float(mean_absolute_error(y[test_idx], fold_prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y[test_idx], fold_prediction))),
        })

    valid = np.isfinite(predictions)

    metrics = {
        "experiment": experiment_name,
        "n_images": int(valid.sum()),
        "n_features": int(len(feature_columns)),
        "mae": float(mean_absolute_error(y[valid], predictions[valid])),
        "rmse": float(np.sqrt(mean_squared_error(y[valid], predictions[valid]))),
        "r2": float(r2_score(y[valid], predictions[valid])),
        "validation": "B1 Leave-One-Concentration-Group-Out",
    }

    output_columns = [
        column for column in (
            "batch", "concentration", "dilution_percent",
            "zoom", "file_name", "relative_path"
        ) if column in data.columns
    ]
    prediction_data = data[output_columns].copy()
    prediction_data["experiment"] = experiment_name
    prediction_data["oof_prediction"] = predictions
    prediction_data["absolute_error"] = np.abs(y - predictions)

    return metrics, prediction_data, pd.DataFrame(fold_rows)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}\n"
            "Please check that unified_all_batches.csv exists."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "batch", "concentration", "dilution_percent", "relative_path"
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    print(f"Loaded unified dataset: {len(df)} rows")
    print("Creating V6 central elliptical ROI features...")

    feature_rows = []
    audit_rows = []
    preview_counts = {}

    for row_index, row in df.iterrows():
        image_path = resolve_image_path(
            row["relative_path"],
            row["batch"],
            row["concentration"],
        )

        audit = {
            "row_index": int(row_index),
            "batch": row["batch"],
            "concentration": row["concentration"],
            "dilution_percent": row["dilution_percent"],
            "zoom": row["zoom"] if "zoom" in df.columns else "",
            "relative_path": row["relative_path"],
            "image_path": str(image_path) if image_path else "",
            "status": "failed",
            "reason": "",
        }

        try:
            if image_path is None:
                raise FileNotFoundError(
                    "Image path could not be resolved; "
                    "checked B1 concentration-relative and B2 raw-relative paths"
                )

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV could not read image: {image_path}")

            image = resize_image(image)
            height, width = image.shape[:2]

            base_metadata = {
                "batch": row["batch"],
                "concentration": row["concentration"],
                "dilution_percent": float(row["dilution_percent"]),
                "zoom": row["zoom"] if "zoom" in df.columns else np.nan,
                "file_name": Path(str(row["relative_path"])).name,
                "relative_path": row["relative_path"],
            }

            for percentage in ROI_SIZES:
                mask = make_center_ellipse_mask(height, width, percentage)
                features = extract_roi_features(image, mask)

                feature_row = dict(base_metadata)
                feature_row["roi_size"] = percentage
                feature_row.update(features)
                feature_rows.append(feature_row)

                group_key = (
                    str(row["batch"]),
                    str(row["concentration"]),
                    str(row["zoom"]) if "zoom" in df.columns else "NA",
                    percentage,
                )
                preview_count = preview_counts.get(group_key, 0)

                if preview_count < PREVIEWS_PER_GROUP_PER_SIZE:
                    safe_batch = str(row["batch"]).replace("/", "_")
                    safe_concentration = str(row["concentration"]).replace("/", "_")
                    safe_zoom = str(row["zoom"]) if "zoom" in df.columns else "NA"
                    preview_name = (
                        f"{safe_batch}_{safe_concentration}_{safe_zoom}_"
                        f"{preview_count + 1}_{percentage}.jpg"
                    )
                    save_preview(
                        image,
                        mask,
                        percentage,
                        PREVIEW_DIR / preview_name,
                    )
                    preview_counts[group_key] = preview_count + 1

            audit["status"] = "processed"

        except Exception as error:
            audit["reason"] = f"{type(error).__name__}: {error}"

        audit_rows.append(audit)

        if (row_index + 1) % 100 == 0:
            print(f"Processed source images: {row_index + 1}/{len(df)}")

    features_df = pd.DataFrame(feature_rows)
    audit_df = pd.DataFrame(audit_rows)

    features_path = OUT_ROOT / "v6_roi_features.csv"
    audit_path = OUT_ROOT / "v6_roi_audit.csv"

    features_df.to_csv(features_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    print("\nROI extraction summary")
    print(audit_df["status"].value_counts(dropna=False).to_string())
    print("\nStatus by batch:")
    print(pd.crosstab(audit_df["batch"], audit_df["status"]).to_string())

    expected_rows = len(df) * len(ROI_SIZES)
    print(f"\nFeature rows: {len(features_df)} (expected {expected_rows})")
    print(f"Feature CSV: {features_path}")
    print(f"Audit CSV: {audit_path}")
    print(f"Preview folder: {PREVIEW_DIR}")

    # B1 only for grouped cross-validation.
    b1 = df[df["batch"].astype(str).str.upper().eq("B1")].copy()

    full_metadata = {
        "batch", "concentration", "dilution_percent", "zoom",
        "file_name", "relative_path", "format", "camera_mode",
        "image_width", "image_height", "width", "height",
    }
    full_feature_columns = get_feature_columns(df, full_metadata)

    results = []
    prediction_frames = []
    fold_frames = []

    if len(b1) == 0:
        print("WARNING: No B1 rows found; model comparison skipped.")
    elif full_feature_columns:
        metrics, predictions, folds = evaluate_logo(
            b1, full_feature_columns, "FULL_IMAGE_BASELINE"
        )
        results.append(metrics)
        prediction_frames.append(predictions)
        fold_frames.append(folds)
    else:
        print("WARNING: No numeric full-image baseline features found.")

    # Evaluate each ROI size only if every B1 source image has its features.
    for percentage in ROI_SIZES:
        roi_subset = features_df[
            (features_df["batch"].astype(str).str.upper() == "B1")
            & (features_df["roi_size"] == percentage)
        ].copy()

        roi_metadata = {
            "batch", "concentration", "dilution_percent", "zoom",
            "file_name", "relative_path", "roi_size",
        }
        roi_feature_columns = get_feature_columns(
            roi_subset, roi_metadata
        )

        if len(roi_subset) != len(b1):
            print(
                f"WARNING: ROI {percentage}% has {len(roi_subset)} B1 rows; "
                f"expected {len(b1)}. Skipping this size."
            )
            continue

        metrics, predictions, folds = evaluate_logo(
            roi_subset,
            roi_feature_columns,
            f"CENTRAL_ELLIPSE_{percentage}PCT",
        )
        results.append(metrics)
        prediction_frames.append(predictions)
        fold_frames.append(folds)

    results_df = pd.DataFrame(results)
    results_path = MODEL_DIR / "v6_model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    if prediction_frames:
        all_predictions = pd.concat(prediction_frames, ignore_index=True)
        all_predictions.to_csv(
            MODEL_DIR / "v6_b1_oof_predictions.csv", index=False
        )

        concentration_errors = (
            all_predictions.groupby(
                ["experiment", "concentration"], dropna=False
            )
            .agg(
                n=("absolute_error", "size"),
                mae=("absolute_error", "mean"),
                median_absolute_error=("absolute_error", "median"),
            )
            .reset_index()
        )
        concentration_errors.to_csv(
            MODEL_DIR / "v6_concentration_wise_errors.csv",
            index=False,
        )

    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(
            MODEL_DIR / "v6_fold_metrics.csv",
            index=False,
        )

    # B2 is exploratory only; it has already been used in prior evaluation.
    try:
        b2 = df[df["batch"].astype(str).str.upper().eq("B2")].copy()

        if len(b1) and len(b2) and full_feature_columns:
            X_train = b1[full_feature_columns].replace(
                [np.inf, -np.inf], np.nan
            )
            y_train = b1["dilution_percent"].astype(float)
            X_test = b2[full_feature_columns].replace(
                [np.inf, -np.inf], np.nan
            )

            model = build_model()
            model.fit(X_train, y_train)
            b2_predictions = model.predict(X_test)

            b2_output_columns = [
                column for column in (
                    "batch", "concentration", "dilution_percent",
                    "zoom", "file_name", "relative_path"
                ) if column in b2.columns
            ]
            b2_output = b2[b2_output_columns].copy()
            b2_output["experiment"] = (
                "FULL_IMAGE_B1_TRAIN_B2_EXPLORATORY"
            )
            b2_output["prediction"] = b2_predictions
            b2_output["absolute_error"] = np.abs(
                b2_output["dilution_percent"].to_numpy()
                - b2_predictions
            )
            b2_output.to_csv(
                MODEL_DIR / "v6_b2_exploratory_predictions.csv",
                index=False,
            )
    except Exception as error:
        print(f"B2 exploratory evaluation skipped: {error}")

    print("\nV6 MODEL COMPARISON")
    if not results_df.empty:
        print(results_df.to_string(index=False))
    else:
        print("No model comparison results were produced.")

    print(f"\nComparison report: {results_path}")
    print(f"All V6 outputs: {OUT_ROOT}")
    print(
        "\nIMPORTANT: This experiment tests fixed central ROIs, "
        "not true milk segmentation."
    )
    print(
        "Do not replace the existing baseline unless results and previews "
        "justify it."
    )


if __name__ == "__main__":
    main()
