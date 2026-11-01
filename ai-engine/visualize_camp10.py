
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

MANIFEST = (
    HERE / "models" / "sp_milk_hog_svm_10campaigns_v3_corrected"
    / "manifest_with_splits.csv"
)

OUTPUT = HERE / "analysis_output" / "camp10_visual_comparison.png"

df = pd.read_csv(MANIFEST)

selected_categories = ["RAD", "SAD1", "R", "S1"]

subset = df[
    (df["campaign"] == "CAMP10")
    & (df["category_corrected"].isin(selected_categories))
]

fig, axes = plt.subplots(4, 3, figsize=(13, 15))

for row, category in enumerate(selected_categories):
    category_df = subset[
        subset["category_corrected"] == category
    ].sort_values("file_path")

    samples = category_df.head(3)

    if len(samples) != 3:
        raise ValueError(f"Expected 3 images for {category}")

    for col, (_, sample) in enumerate(samples.iterrows()):
        path = Path(sample["file_path"])
        if not path.is_absolute():
            path = HERE / path

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot read image: {path}")

        axes[row, col].imshow(image, cmap="gray")
        axes[row, col].set_title(
            f"{category} | {sample['class_name']}\n"
            f"Frame {path.stem.split('_frame_')[-1]}"
        )
        axes[row, col].axis("off")

plt.suptitle(
    "CAMP10 Laser Speckle Image Comparison",
    fontsize=16
)

plt.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")

print("Comparison saved:", OUTPUT)

plt.show()
