
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from skimage.feature import local_binary_pattern

HERE = Path(__file__).resolve().parent

MANIFEST = (
    HERE / "models" / "sp_milk_hog_svm_10campaigns_v3_corrected"
    / "manifest_with_splits.csv"
)

OUTPUT = HERE / "analysis_output" / "camp10_texture_statistics.csv"

df = pd.read_csv(MANIFEST)

df = df[
    (df["campaign"] == "CAMP10")
].copy()

# Same number of images from each class
df = (
    df.groupby("class_name", group_keys=False)
    .sample(n=300, random_state=42)
    .reset_index(drop=True)
)

results = []

for i, row in df.iterrows():

    path = Path(str(row["file_path"]))
    if not path.is_absolute():
        path = HERE / path

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Skipped:", path)
        continue

    # Resize for consistent statistical comparison
    image = cv2.resize(image, (512, 512))

    image_float = image.astype(np.float32)

    lbp = local_binary_pattern(
        image, P=8, R=1, method="uniform"
    )

    results.append({
        "file_path": str(path),
        "category": row["category_corrected"],
        "class_name": row["class_name"],
        "mean_brightness": float(np.mean(image_float)),
        "brightness_std": float(np.std(image_float)),
        "contrast": float(image_float.std()),
        "entropy": float(
            -np.sum(
                (lambda h: h[h > 0] * np.log2(h[h > 0]))(
                    np.histogram(image, bins=256, range=(0, 256), density=True)[0]
                )
            )
        ),
        "sharpness": float(
            cv2.Laplacian(image, cv2.CV_64F).var()
        ),
        "lbp_mean": float(np.mean(lbp)),
        "lbp_std": float(np.std(lbp))
    })

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(df)}")

result_df = pd.DataFrame(results)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT, index=False)

print("\nClass-wise feature comparison:")
print(
    result_df.groupby("class_name")
    .mean(numeric_only=True)
    .round(3).to_string()
)

print("\nSaved:", OUTPUT)
