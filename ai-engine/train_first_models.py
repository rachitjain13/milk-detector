
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

feature_columns = [
    c for c in data.select_dtypes(include="number").columns
    if c not in exclude
]

X = data[feature_columns]
y = data["water_added_pct"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300, random_state=42
    ),
    "SVR": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVR(kernel="rbf", C=10, epsilon=1.0)
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })

    print(f"\n{name}")
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 3))

results_df = pd.DataFrame(results)
results_df.to_csv(
    "analysis_output/first_model_results.csv", index=False
)

print("\nResults saved: analysis_output/first_model_results.csv")
    