
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

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
groups = data["sample_id"]

model = ExtraTreesRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2
)

cv = GroupKFold(n_splits=5)

predictions = cross_val_predict(
    model, X, y, groups=groups, cv=cv
)

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

print("===== GROUP VALIDATION RESULTS =====")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2:", round(r2, 3))

comparison = pd.DataFrame({
    "sample_id": groups,
    "actual": y,
    "predicted": predictions
})

print("\nAverage predictions by sample:")
print(
    comparison.groupby("sample_id")
    [["actual", "predicted"]]
    .mean().round(3)
)

comparison.to_csv(
    "analysis_output/group_validation_predictions.csv",
    index=False
)

print("\nSaved: analysis_output/group_validation_predictions.csv")
