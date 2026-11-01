import csv
from pathlib import Path
from collections import Counter

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "research_data" / "sp_milk"
SPLIT_FILE = DATA / "camp1_split.csv"
MODEL_DIR = DATA / "models"
MODEL_FILE = MODEL_DIR / "camp1_hog_svm.joblib"

IMAGE_SIZE = (128, 128)
LABELS = ["adulterated", "unadulterated"]

if not SPLIT_FILE.is_file():
    raise SystemExit(f"Split CSV not found: {SPLIT_FILE}")

with SPLIT_FILE.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

def extract_features(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    image = cv2.resize(image, IMAGE_SIZE)
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features

features_by_split = {}
labels_by_split = {}

for split in ("train", "validation", "test"):
    subset = [r for r in rows if r["split"].strip() == split]
    if not subset:
        raise SystemExit(f"No rows found for split: {split}")

    X, y = [], []
    for row in subset:
        label = row["label"].strip().lower()
        if label not in LABELS:
            raise SystemExit(f"Unexpected label: {label}")

        X.append(extract_features(row["image_path"]))
        y.append(label)

    features_by_split[split] = np.asarray(X, dtype=np.float32)
    labels_by_split[split] = np.asarray(y)

    print(f"{split}: {len(y)} images | {dict(Counter(y))}")

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", class_weight="balanced"),
)

print("\nTraining HOG + SVM...")
model.fit(features_by_split["train"], labels_by_split["train"])

for split in ("validation", "test"):
    X = features_by_split[split]
    y_true = labels_by_split[split]
    y_pred = model.predict(X)

    print(f"\n===== {split.upper()} RESULTS =====")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "model": model,
    "image_size": IMAGE_SIZE,
    "labels": LABELS,
    "feature_type": "HOG",
}, MODEL_FILE)

print(f"\nModel saved to: {MODEL_FILE}")
print("Baseline training complete.")
print("Note: CAMP1 contains only 6 independent samples per class.")
