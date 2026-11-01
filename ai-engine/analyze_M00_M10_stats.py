import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind

files = {
    "M00": Path("analysis_output/M00_phone_features.csv"),
    "M10": Path("analysis_output/M10_phone_features.csv"),
}

frames = []

for sample, path in files.items():
    df = pd.read_csv(path)
    df["sample"] = sample
    df["zoom"] = df["relative_path"].str.split("\\").str[0]
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

selected_features = [
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

results = []

for zoom in ["0.5", "1.0"]:
    subset = data[data["zoom"] == zoom]

    for feature in selected_features:
        a = subset[subset["sample"] == "M00"][feature].dropna()
        b = subset[subset["sample"] == "M10"][feature].dropna()

        if len(a) < 2 or len(b) < 2:
            continue

        test = ttest_ind(a, b, equal_var=False)

        results.append({
            "zoom": zoom,
            "feature": feature,
            "M00_mean": a.mean(),
            "M10_mean": b.mean(),
            "M00_std": a.std(),
            "M10_std": b.std(),
            "difference": b.mean() - a.mean(),
            "p_value": test.pvalue,
        })

result_df = pd.DataFrame(results)
result_df["abs_difference"] = result_df["difference"].abs()
result_df = result_df.sort_values(
    ["zoom", "abs_difference"], ascending=[True, False]
)

output = Path("analysis_output/M00_M10_statistical_analysis.csv")
result_df.to_csv(output, index=False)

print(result_df.round(5).to_string(index=False))
print(f"\nSaved: {output}")