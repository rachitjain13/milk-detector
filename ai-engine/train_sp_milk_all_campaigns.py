"""Train a HOG + SVM baseline on the SP-MILK laser-speckle dataset.

Place this file in MilkOptic-AI/ai-engine/ and run:
    python train_sp_milk_all_campaigns.py

Expected folders:
    ai-engine/research_data/sp_milk/extracted/CAMP1/...
    ai-engine/research_data/sp_milk/extracted/CAMP1_UNADULTERATED/...
    ...

The script splits by campaign (not random individual images) to reduce leakage.
"""
from pathlib import Path
import json
import re
import sys
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "research_data" / "sp_milk" / "extracted"
OUTPUT_DIR = HERE / "models" / "sp_milk_hog_svm"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = (128, 128)
LABEL_NAMES = {0: "adulterated", 1: "unadulterated"}


def find_images():
    rows = []
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_ROOT}")

    for class_dir in sorted(DATA_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue
        name = class_dir.name.upper()
        unad_match = re.fullmatch(r"CAMP(\d+)_UNADULTERATED", name)
        ad_match = re.fullmatch(r"CAMP(\d+)", name)
        if unad_match:
            campaign = int(unad_match.group(1))
            label = 1
        elif ad_match:
            campaign = int(ad_match.group(1))
            label = 0
        else:
            continue

        files = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        for path in files:
            rows.append({"path": str(path.resolve()), "label": label,
                         "class_name": LABEL_NAMES[label], "campaign": campaign,
                         "source_folder": class_dir.name})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {DATA_ROOT}. Check extracted folder names.")
    return df.drop_duplicates(subset=["path"]).sort_values(["campaign", "label", "path"]).reset_index(drop=True)


def assign_campaign_splits(df):
    campaigns = sorted(df["campaign"].unique().tolist())
    if len(campaigns) < 3:
        raise RuntimeError(
            f"Need at least 3 campaigns for a campaign-level train/validation/test split; found {campaigns}. "
            "Extract more campaigns before training."
        )

    # Preferred split when all 10 campaigns are available: campaigns 1-7 train, 8-9 validation, 10 test.
    if set(range(1, 11)).issubset(set(campaigns)):
        train_camps, val_camps, test_camps = set(range(1, 8)), {8, 9}, {10}
    else:
        # For a partial dataset, reserve the latest campaigns for validation and test.
        n = len(campaigns)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        if n_test + n_val >= n:
            n_test, n_val = 1, 1
        train_camps = set(campaigns[: n - n_val - n_test])
        val_camps = set(campaigns[n - n_val - n_test : n - n_test])
        test_camps = set(campaigns[n - n_test :])

    def split(c):
        if c in train_camps:
            return "train"
        if c in val_camps:
            return "validation"
        if c in test_camps:
            return "test"
        return "unused"

    out = df.copy()
    out["split"] = out["campaign"].map(split)
    return out, sorted(train_camps), sorted(val_camps), sorted(test_camps)


def image_to_hog(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                   block_norm="L2-Hys", transform_sqrt=True, feature_vector=True)
    return features.astype(np.float32)


def load_features(df, split_name):
    part = df[df["split"] == split_name].reset_index(drop=True)
    X, y, kept_paths = [], [], []
    bad = []
    print(f"\nLoading {split_name}: {len(part)} images...", flush=True)
    for i, row in part.iterrows():
        try:
            X.append(image_to_hog(row["path"]))
            y.append(int(row["label"]))
            kept_paths.append(row["path"])
        except Exception as exc:
            bad.append((row["path"], str(exc)))
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(part)}", flush=True)
    if bad:
        print(f"Warning: skipped {len(bad)} unreadable files in {split_name}.")
    if not X:
        raise RuntimeError(f"No readable images in split {split_name}.")
    return np.vstack(X), np.asarray(y, dtype=np.int32), kept_paths


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset root: {DATA_ROOT}")
    df = find_images()
    df, train_camps, val_camps, test_camps = assign_campaign_splits(df)
    df.to_csv(OUTPUT_DIR / "image_manifest_with_splits.csv", index=False)

    print("\nFound images by class:")
    print(df.groupby(["campaign", "class_name"]).size().unstack(fill_value=0).to_string())
    print("\nCampaign split:")
    print(f"  Train campaigns: {train_camps}")
    print(f"  Validation campaigns: {val_camps}")
    print(f"  Test campaigns: {test_camps}")
    print("\nImage counts by split/class:")
    print(df[df["split"] != "unused"].groupby(["split", "class_name"]).size().unstack(fill_value=0).to_string())

    # Do not silently continue if a split is missing one class.
    for split_name in ("train", "validation", "test"):
        labels = set(df.loc[df["split"] == split_name, "label"].tolist())
        if labels != {0, 1}:
            raise RuntimeError(f"Split '{split_name}' must contain both classes; found labels {labels}.")

    X_train, y_train, _ = load_features(df, "train")
    X_val, y_val, _ = load_features(df, "validation")
    X_test, y_test, _ = load_features(df, "test")

    model = make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", random_state=42, max_iter=20000))
    print("\nTraining LinearSVC...", flush=True)
    model.fit(X_train, y_train)

    results = {}
    for split_name, X, y in (("validation", X_val, y_val), ("test", X_test, y_test)):
        pred = model.predict(X)
        report = classification_report(y, pred, labels=[0, 1], target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        acc = float(accuracy_score(y, pred))
        results[split_name] = {"accuracy": acc, "confusion_matrix_labels_order": ["adulterated", "unadulterated"],
                               "confusion_matrix": cm, "classification_report": report,
                               "num_images": int(len(y))}
        print(f"\n{split_name.upper()} accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print("Confusion matrix [rows=true, cols=predicted], order [adulterated, unadulterated]:")
        print(np.array(cm))
        print(classification_report(y, pred, labels=[0, 1], target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], zero_division=0))

    model_path = OUTPUT_DIR / "sp_milk_hog_svm.joblib"
    joblib.dump({"model": model, "image_size": IMAGE_SIZE,
                 "feature": "HOG grayscale; orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)",
                 "labels": LABEL_NAMES, "train_campaigns": train_camps,
                 "validation_campaigns": val_camps, "test_campaigns": test_camps}, model_path)
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {OUTPUT_DIR / 'metrics.json'}")
    print(f"Saved manifest: {OUTPUT_DIR / 'image_manifest_with_splits.csv'}")
    print("\nReminder: this model is trained for SP-MILK laser-speckle images, not ordinary phone-camera milk photos.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
