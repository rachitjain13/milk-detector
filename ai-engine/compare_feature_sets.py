
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("analysis_output/milkoptic_ml_dataset.csv")

top_features = [
    "hsv_v_p10",
    "rgb_g_p10",
    "rgb_b_std",
    "gray_hist_bin_04",
    "rgb_r_p10",
    "rgb_b_p10",
    "image_brightness_std_gray",
    "rgb_r_std",
    "rgb_g_std",
    "hsv_s_std",
]

exclude = [
    "file_name",
    "relative_path",
    "zoom",
    "sample_id",
    "water_added_pct",
    "image_width_original",
    "image_height_original",
]

all_features = [
    c for c in data.select_dtypes(include="number").columns
    if c not in exclude
]

y = data["water_added_pct"]
groups = data["sample_id"]

cv = GroupKFold(n_splits=5)

for name, features in [
    ("All 84 features", all_features),
    ("Top 10 features", top_features),
]:
    X = data[features]

    model = ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=2
    )

    predictions = cross_val_predict(
        model, X, y, groups=groups, cv=cv
    )

    print(f"\n===== {name} =====")
    print("MAE:", round(mean_absolute_error(y, predictions), 3))
    print("RMSE:", round(np.sqrt(mean_squared_error(y, predictions)), 3))
    print("R2:", round(r2_score(y, predictions), 3))

print("\nFeature comparison completed.")
