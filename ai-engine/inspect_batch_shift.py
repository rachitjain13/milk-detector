import pandas as pd

df = pd.read_csv(
    "analysis_output/b1_b2_feature_distribution.csv"
)

features = [
    "image_brightness_std_gray",
    "hsv_v_std",
    "rgb_g_std",
    "rgb_b_std",
    "rgb_r_std"
]

for feature in features:
    d = df[df["feature"] == feature]

    print("\n" + "=" * 70)
    print("FEATURE:", feature)

    print(
        d[[
            "concentration",
            "zoom",
            "B1_mean",
            "B2_mean",
            "B1_std",
            "B2_std",
            "standardized_difference"
        ]].round(4).to_string(index=False)
    )
