from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
manifest = pd.read_csv(
    ROOT / "analysis_output" / "phone_rgb_batch_manifest.csv"
)

fig, axes = plt.subplots(5, 4, figsize=(16, 18))

for row, concentration in enumerate(
    ["M00", "M10", "M20", "M30", "M40"]
):
    for col, (batch, zoom) in enumerate([
        ("B1", 0.5),
        ("B2", 0.5),
        ("B1", 1.0),
        ("B2", 1.0)
    ]):
        subset = manifest[
            (manifest["batch"] == batch) &
            (manifest["concentration"] == concentration) &
            (manifest["zoom"] == zoom)
        ]

        path = Path(subset.iloc[len(subset)//2]["file_path"])
        image = cv2.imread(str(path))

        if image is None:
            axes[row, col].set_title("Image unreadable")
            axes[row, col].axis("off")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[row, col].imshow(image)
        axes[row, col].set_title(
            f"{batch} | {concentration} | {zoom}x"
        )
        axes[row, col].axis("off")

plt.tight_layout()

output = ROOT / "analysis_output" / "batch_visual_comparison.png"
plt.savefig(output, dpi=150)
plt.close()

print("Saved:", output)
