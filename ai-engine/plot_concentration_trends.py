import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

samples = {
    "M00": "analysis_output/M00_phone_features.csv",
    "M10": "analysis_output/M10_phone_features.csv",
    "M20": "analysis_output/M20_phone_features.csv",
}

features = [
    "image_mean_brightness_gray",
    "rgb_r_mean",
    "rgb_g_mean",
    "rgb_b_mean",
    "hsv_s_mean",
    "glcm_contrast_mean",
    "glcm_homogeneity_mean",
    "glcm_energy_mean",
]

rows = []

for sample, file in samples.items():
    df = pd.read_csv(file)
    df["zoom"] = df["relative_path"].str.split("\\").str[0]
    df = df[df["zoom"] == "1.0"]

    row = {"sample": sample}
    for feature in features:
        row[feature] = df[feature].mean()

    rows.append(row)

summary = pd.DataFrame(rows)
summary["water_added_pct"] = [0, 10, 20]

for feature in features:
    plt.figure(figsize=(7, 4))
    plt.plot(
        summary["water_added_pct"],
        summary[feature],
        marker="o"
    )
    plt.xlabel("Water added (% of starting milk volume)")
    plt.ylabel(feature)
    plt.title(f"Concentration trend: {feature}")
    plt.grid(True)

    output = Path("analysis_output") / f"trend_{feature}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()

print("Trend graphs saved in analysis_output/")