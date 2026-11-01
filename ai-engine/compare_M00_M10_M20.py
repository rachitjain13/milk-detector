import pandas as pd
from pathlib import Path

files = {
    "M00": "analysis_output/M00_phone_features.csv",
    "M10": "analysis_output/M10_phone_features.csv",
    "M20": "analysis_output/M20_phone_features.csv",
}

frames = []

for sample, file in files.items():
    df = pd.read_csv(file)
    df["sample"] = sample
    df["zoom"] = df["relative_path"].str.split("\\").str[0]
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

features = [
    "image_mean_brightness_gray",
    "image_brightness_std_gray",
    "image_blur_laplacian_var",
    "image_edge_density",
    "rgb_r_mean",
    "rgb_g_mean",
    "rgb_b_mean",
    "hsv_s_mean",
    "hsv_v_mean",
    "glcm_contrast_mean",
    "glcm_homogeneity_mean",
    "glcm_energy_mean",
]

summary = data.groupby(["zoom", "sample"])[features].agg(
    ["mean", "std"]
).round(3)

print("\n===== M00 vs M10 vs M20 =====")

for zoom in sorted(data["zoom"].dropna().unique()):
    print(f"\n========== {zoom} CAMERA ==========")
    print(summary.loc[zoom].to_string())

output = Path("analysis_output/M00_M10_M20_comparison.csv")
summary.to_csv(output)

print(f"\nComparison saved: {output}")