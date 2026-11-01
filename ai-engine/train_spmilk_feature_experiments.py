
from pathlib import Path
import json
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd

from skimage.feature import hog, local_binary_pattern
from skimage.measure import shannon_entropy

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent

MANIFEST = HERE / "analysis_output" / "spmilk_master_manifest.csv"
OUTPUT = HERE / "models" / "sp_milk_feature_experiments"

IMAGE_SIZE = (128, 128)

# Manifest: 0 = UNADULTERATED, 1 = ADULTERATED
LABEL_NAMES = ["UNADULTERATED", "ADULTERATED"]

PAIRS = {
    "HOG": 8100,
    "INTENSITY": None,
    "LBP": None,
    "COMBINED": None
}


def intensity_features(image):
    img = image.astype(np.float32)

    return np.array([
        np.mean(img),
        np.std(img),
        np.percentile(img, 10),
        np.percentile(img, 25),
        np.percentile(img, 50),
        np.percentile(img, 75),
        np.percentile(img, 90),
        np.max(img) - np.min(img),
        shannon_entropy(image),
        cv2.Laplacian(image, cv2.CV_64F).var()
    ], dtype=np.float32)


def lbp_features(image):
    lbp = local_binary_pattern(
        image, P=8, R=1, method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 12),
        density=True
    )

    return hist.astype(np.float32)


def extract_all_features(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    hog_feature = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    ).astype(np.float32)

    intensity = intensity_features(image)
    lbp = lbp_features(image)

    return {
        "HOG": hog_feature,
        "INTENSITY": intensity,
        "LBP": lbp,
        "COMBINED": np.concatenate([
            hog_feature, intensity, lbp
        ])
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)

    required = {
        "file_path", "campaign", "label", "class_name"
    }

    if not required.issubset(df.columns):
        raise ValueError("Master manifest columns are missing.")

    df["campaign_num"] = (
        df["campaign"].astype(str)
        .str.extract(r"(\d+)")[0].astype(int)
    )

    df["absolute_path"] = df["file_path"].apply(
        lambda p: Path(p) if Path(p).is_absolute()
        else HERE / p
    )

    if len(df) != 12000:
        raise ValueError(f"Expected 12000 images, found {len(df)}")

    if df["absolute_path"].duplicated().any():
        raise ValueError("Duplicate image paths found.")

    if not df["absolute_path"].apply(Path.exists).all():
        raise FileNotFoundError("Some manifest image paths are missing.")

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Expected labels 0 and 1.")

    df["split"] = df["campaign_num"].map(
        lambda c: "train" if c <= 7
        else "validation" if c in (8, 9)
        else "test"
    )

    print("Dataset split:")
    print(pd.crosstab(df["split"], df["class_name"]))

    feature_data = {
        name: {"X": [], "y": []}
        for name in PAIRS
    }

    failures = []

    print("\nExtracting features...")

    for i, row in df.iterrows():
        try:
            features = extract_all_features(row["absolute_path"])

            for name in PAIRS:
                feature_data[name]["X"].append(features[name])
                feature_data[name]["y"].append(int(row["label"]))

        except Exception as exc:
            failures.append({
                "file_path": str(row["absolute_path"]),
                "error": str(exc)
            })

        if (i + 1) % 250 == 0:
            print(f"Processed {i + 1}/{len(df)}", flush=True)

    if failures:
        pd.DataFrame(failures).to_csv(
            OUTPUT / "feature_failures.csv", index=False
        )
        raise RuntimeError(
            f"{len(failures)} images failed. Check feature_failures.csv"
        )

    split_masks = {
        name: df["split"].to_numpy() == name
        for name in ["train", "validation", "test"]
    }

    results = {}

    for feature_name, data in feature_data.items():
        print(f"\nTraining {feature_name}...")

        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int32)

        X_train = X[split_masks["train"]]
        y_train = y[split_masks["train"]]

        X_val = X[split_masks["validation"]]
        y_val = y[split_masks["validation"]]

        X_test = X[split_masks["test"]]
        y_test = y[split_masks["test"]]

        model = make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                random_state=42,
                max_iter=20000
            )
        )

        model.fit(X_train, y_train)

        results[feature_name] = {}

        for split_name, features, labels in [
            ("validation", X_val, y_val),
            ("test", X_test, y_test)
        ]:
            pred = model.predict(features)

            results[feature_name][split_name] = {
                "accuracy": float(accuracy_score(labels, pred)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, pred)
                ),
                "confusion_matrix": confusion_matrix(
                    labels, pred, labels=[0, 1]
                ).tolist(),
                "classification_report": classification_report(
                    labels,
                    pred,
                    labels=[0, 1],
                    target_names=LABEL_NAMES,
                    output_dict=True,
                    zero_division=0
                )
            }

            print(
                split_name,
                "Accuracy:", round(
                    results[feature_name][split_name]["accuracy"], 4
                ),
                "Balanced accuracy:", round(
                    results[feature_name][split_name]["balanced_accuracy"], 4
                )
            )

        if feature_name == "COMBINED":
            joblib.dump(
                model,
                OUTPUT / "combined_hog_intensity_lbp_model.joblib"
            )

        del X, X_train, X_val, X_test, model

    with open(OUTPUT / "comparison_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nFINAL COMPARISON")
    for name, scores in results.items():
        print(
            name,
            "| Validation:",
            round(scores["validation"]["balanced_accuracy"], 4),
            "| CAMP10 Test:",
            round(scores["test"]["balanced_accuracy"], 4)
        )

    print("\nAll experiments completed.")
    print("Results:", OUTPUT / "comparison_metrics.json")


if __name__ == "__main__":
    main()
