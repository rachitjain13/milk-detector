
from pathlib import Path
import json
import sys

import cv2
import joblib
import numpy as np
import pandas as pd

from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent

MANIFEST = HERE / "analysis_output" / "spmilk_master_manifest.csv"
OUTPUT_DIR = HERE / "models" / "sp_milk_hog_svm_10campaigns_v3_corrected"

IMAGE_SIZE = (128, 128)

LABEL_NAMES = {
    0: "UNADULTERATED",
    1: "ADULTERATED"
}


def load_manifest():
    df = pd.read_csv(MANIFEST)

    required = {
        "file_path", "campaign", "jug", "label", "class_name"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")

    df["campaign_num"] = (
        df["campaign"].astype(str).str.extract(r"(\d+)").astype(int)
    )

    # Correct category from folder path:
    # .../CAMP1/SAD5/CUV1/image.png -> SAD5
    def get_category(path):
        parts = Path(str(path).replace("\\", "/")).parts
        for i, part in enumerate(parts):
            if part.upper().startswith("CAMP") and i + 1 < len(parts):
                next_part = parts[i + 1]
                if next_part.upper() != part.upper():
                    return next_part
        return "UNKNOWN"

    df["category_corrected"] = df["file_path"].apply(get_category)

    # Resolve relative paths from ai-engine directory.
    def resolve_path(p):
        path = Path(str(p))
        return path if path.is_absolute() else HERE / path

    df["absolute_path"] = df["file_path"].apply(resolve_path)

    missing_files = df[~df["absolute_path"].apply(Path.exists)]
    if len(missing_files):
        raise FileNotFoundError(
            f"{len(missing_files)} manifest image paths do not exist. "
            f"Example: {missing_files.iloc[0]['absolute_path']}"
        )

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Expected labels 0 and 1.")

    if df["absolute_path"].duplicated().any():
        raise ValueError("Duplicate image paths found in manifest.")

    return df


def assign_splits(df):
    campaigns = set(df["campaign_num"].unique())

    required = set(range(1, 11))
    if not required.issubset(campaigns):
        raise ValueError(
            f"Expected campaigns 1-10. Found: {sorted(campaigns)}"
        )

    df = df.copy()

    df["split"] = df["campaign_num"].map(
        lambda c: (
            "train" if c <= 7
            else "validation" if c in (8, 9)
            else "test"
        )
    )

    for split in ("train", "validation", "test"):
        labels = set(df.loc[df["split"] == split, "label"])
        if labels != {0, 1}:
            raise ValueError(
                f"{split} split does not contain both classes: {labels}"
            )

    return df


def image_to_hog(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )

    return features.astype(np.float32)


def extract_features(df, split_name):
    part = df[df["split"] == split_name].reset_index(drop=True)

    X, y = [], []
    failed = []

    print(f"\nExtracting {split_name}: {len(part)} images", flush=True)

    for i, row in part.iterrows():
        try:
            X.append(image_to_hog(row["absolute_path"]))
            y.append(int(row["label"]))
        except Exception as exc:
            failed.append({
                "file_path": str(row["absolute_path"]),
                "error": str(exc)
            })

        if (i + 1) % 250 == 0:
            print(f"Processed {i + 1}/{len(part)}", flush=True)

    if failed:
        print(f"WARNING: {len(failed)} images failed.")

    if not X:
        raise RuntimeError(f"No features extracted for {split_name}")

    return np.vstack(X), np.asarray(y), failed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading verified master manifest...")
    df = load_manifest()
    df = assign_splits(df)

    print(f"\nTotal images: {len(df)}")
    print("\nCampaign distribution:")
    print(
        df.groupby(["campaign", "class_name"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nSplit distribution:")
    print(
        df.groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
    )

    df.to_csv(
        OUTPUT_DIR / "manifest_with_splits.csv",
        index=False
    )

    X_train, y_train, bad_train = extract_features(df, "train")
    X_val, y_val, bad_val = extract_features(df, "validation")
    X_test, y_test, bad_test = extract_features(df, "test")

    if bad_train or bad_val or bad_test:
        failures = bad_train + bad_val + bad_test
        pd.DataFrame(failures).to_csv(
            OUTPUT_DIR / "feature_extraction_failures.csv",
            index=False
        )
        raise RuntimeError(
            f"{len(failures)} feature extraction failures. "
            "Review failure CSV before training."
        )

    print("\nFeature matrix shapes:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=20000
        )
    )

    print("\nTraining HOG + LinearSVC...")
    model.fit(X_train, y_train)

    results = {}

    for split_name, X, y in (
        ("validation", X_val, y_val),
        ("test", X_test, y_test)
    ):
        predictions = model.predict(X)

        cm = confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        )

        report = classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=[
                LABEL_NAMES[0],
                LABEL_NAMES[1]
            ],
            output_dict=True,
            zero_division=0
        )

        results[split_name] = {
            "accuracy": float(accuracy_score(y, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y, predictions)
            ),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_images": int(len(y))
        }

        print(f"\n{split_name.upper()} RESULTS")
        print("Accuracy:", accuracy_score(y, predictions))
        print("Balanced accuracy:", balanced_accuracy_score(y, predictions))
        print("Confusion matrix [true rows, predicted columns]:")
        print(cm)
        print(
            classification_report(
                y,
                predictions,
                labels=[0, 1],
                target_names=[
                    LABEL_NAMES[0],
                    LABEL_NAMES[1]
                ],
                zero_division=0
            )
        )

    model_path = OUTPUT_DIR / "sp_milk_hog_svm_10campaigns_v2.joblib"

    joblib.dump({
        "model": model,
        "image_size": IMAGE_SIZE,
        "feature": (
            "HOG grayscale; orientations=9; "
            "pixels_per_cell=(8,8); cells_per_block=(2,2); "
            "block_norm=L2-Hys; transform_sqrt=True"
        ),
        "labels": LABEL_NAMES,
        "train_campaigns": list(range(1, 8)),
        "validation_campaigns": [8, 9],
        "test_campaigns": [10]
    }, model_path)

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nTraining completed.")
    print("Model:", model_path)
    print("Metrics:", OUTPUT_DIR / "metrics.json")
    print("Manifest:", OUTPUT_DIR / "manifest_with_splits.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
