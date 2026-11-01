
import pandas as pd
from pathlib import Path

files = {
    "M00": "analysis_output/M00_phone_features.csv",
    "M10": "analysis_output/M10_phone_features.csv",
    "M20": "analysis_output/M20_phone_features.csv",
    "M30": "analysis_output/M30_phone_features.csv",
    "M40": "analysis_output/M40_phone_features.csv",
}

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

frames = []

for sample, file in files.items():
    df = pd.read_csv(file)
    df["sample"] = sample
    df["zoom"] = df["relative_path"].str.split("\\").str[0]
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

for zoom in ["0.5", "1.0"]:
    subset = data[data["zoom"] == zoom]

    summary = subset.groupby("sample")[features].agg(
        ["mean", "std"]
    ).round(4)

    print(f"\n========== {zoom}x CAMERA ==========")
    print(summary.to_string())

    output = Path(
        f"analysis_output/all_concentrations_{zoom.replace('.', '_')}x.csv"
    )
    summary.to_csv(output)
    print(f"\nSaved: {output}")

print("\nTotal images:", len(data))
print("Comparison completed.")
