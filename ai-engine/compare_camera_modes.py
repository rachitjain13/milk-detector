
import pandas as pd

manifest = pd.read_csv(
    "analysis_output/phone_rgb_batch_manifest.csv"
)

# Inspect B1 M20 images
subset = manifest[
    (manifest["batch"] == "B1") &
    (manifest["concentration"] == "M20")
].copy()

print("\n===== M20 CAMERA MODE DISTRIBUTION =====")

print(pd.crosstab(
    subset["zoom"],
    subset["camera_mode"],
    margins=True
))

print("\n===== ALL B1 CAMERA MODE DISTRIBUTION =====")

print(pd.crosstab(
    [manifest["concentration"], manifest["zoom"]],
    manifest["camera_mode"]
))

print("\n===== CAMERA MODE COUNTS =====")
print(subset["camera_mode"].value_counts())
