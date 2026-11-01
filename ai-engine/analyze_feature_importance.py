
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

data = pd.read_csv("analysis_output/milkoptic_ml_dataset.csv")

exclude = [
    "file_name",
    "relative_path",
    "zoom",
    "sample_id",
    "water_added_pct",
    "image_width_original",
    "image_height_original",
]

features = [
    c for c in data.select_dtypes(include="number").columns
    if c not in exclude
]

X = data[features]
y = data["water_added_pct"]

model = ExtraTreesRegressor(
    n_estimators=500,
    random_state=42,
    min_samples_leaf=2
)

model.fit(X, y)

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance", ascending=False
)

print("\n===== TOP 20 IMPORTANT FEATURES =====")
print(importance.head(20).to_string(index=False))

importance.to_csv(
    "analysis_output/feature_importance.csv",
    index=False
)

print("\nSaved: analysis_output/feature_importance.csv")
