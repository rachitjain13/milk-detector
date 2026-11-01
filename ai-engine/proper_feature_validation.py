
import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

data = pd.read_csv("analysis_output/milkoptic_ml_dataset.csv")

exclude = [
    "file_name", "relative_path", "zoom",
    "sample_id", "water_added_pct",
    "image_width_original", "image_height_original"
]

features = [
    c for c in data.select_dtypes(include="number").columns
    if c not in exclude
]

X = data[features]
y = data["water_added_pct"]
groups = data["sample_id"]

logo = LeaveOneGroupOut()
predictions = np.zeros(len(data))

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]

    selector_model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    selector_model.fit(X_train, y_train)

    selector = SelectFromModel(
        selector_model,
        threshold=-np.inf,
        max_features=10,
        prefit=True
    )

    selected = selector.get_support()
    selected_features = X.columns[selected]

    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train[selected_features], y_train)
    predictions[test_idx] = model.predict(
        X_test[selected_features]
    )

    print("Held-out group:", groups.iloc[test_idx].iloc[0])
    print("Selected features:", list(selected_features))

print("\n===== PROPER FEATURE SELECTION RESULTS =====")
print("MAE:", round(mean_absolute_error(y, predictions), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y, predictions)), 3))
print("R2:", round(r2_score(y, predictions), 3))

pd.DataFrame({
    "sample_id": groups,
    "actual": y,
    "predicted": predictions
}).to_csv(
    "analysis_output/proper_feature_validation_predictions.csv",
    index=False
)

print("\nPredictions saved.")
