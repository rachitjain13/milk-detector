from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
CANDIDATES = [
    BASE / 'v6_roi_features(1).csv',
    BASE / 'v6_roi_features.csv',
    BASE / 'analysis_output' / 'milk_segmentation_v6_roi' / 'v6_roi_features.csv',
    BASE / 'research_data' / 'v6_roi_features.csv',
]
CSV = next((p for p in CANDIDATES if p.exists()), None)
if CSV is None:
    raise FileNotFoundError('Cannot find v6_roi_features.csv. Keep it beside this script or in analysis_output/milk_segmentation_v6_roi/.')

OUT = BASE / 'models' / 'roi70_ridge'
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
required = {'batch', 'file_name', 'dilution_percent', 'roi_size', 'zoom'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f'Missing required columns: {sorted(missing)}')

# Reproduce the selected V11-V14 feature set: ROI 70%, B1 only.
df = df[(df['batch'].astype(str) == 'B1') & (pd.to_numeric(df['roi_size'], errors='coerce') == 70)].copy()
if len(df) == 0:
    raise ValueError('No B1 rows found for roi_size=70.')

exclude = {'dilution_percent', 'concentration', 'batch', 'file_name', 'relative_path',
           'file_path', 'camera_mode', 'format', 'zoom', 'roi_size'}
features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
if not features:
    raise ValueError('No numeric model features found.')

X = df[features].replace([np.inf, -np.inf], np.nan)
y = pd.to_numeric(df['dilution_percent'], errors='raise').astype(float)
# Match the grouping used in the V11-V14 run.
groups = df['dilution_percent'].astype(str) + '_zoom_' + df['zoom'].astype(str)
cv = GroupKFold(n_splits=3)
model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10.0))
oof = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=1)
metrics = {
    'n': int(len(y)),
    'features': int(len(features)),
    'B1_grouped_CV_MAE_percentage_points': float(mean_absolute_error(y, oof)),
    'B1_grouped_CV_RMSE_percentage_points': float(mean_squared_error(y, oof) ** 0.5),
    'B1_grouped_CV_R2': float(r2_score(y, oof)),
    'B1_grouped_CV_bias_prediction_minus_actual': float(np.mean(oof - y)),
    'validation_note': 'Internal grouped cross-validation on B1; not independent real-world accuracy.'
}

# Fit final deployable pipeline on all B1 ROI-70 records.
model.fit(X, y)
artifact = {
    'pipeline': model,
    'feature_names': features,
    'feature_set': 'ROI_70',
    'model_name': 'Ridge_scaled',
    'ridge_alpha': 10.0,
    'target': 'water_dilution_percent',
    'roi_size': 70,
    'training_batch': 'B1',
}
joblib.dump(artifact, OUT / 'milkoptic_roi70_ridge.joblib')
(OUT / 'model_metadata.json').write_text(json.dumps({**metrics, **{k:v for k,v in artifact.items() if k not in {'pipeline','feature_names'}}, 'feature_names': features}, indent=2))
pd.DataFrame({'actual': y.to_numpy(), 'oof_prediction': oof, 'error': oof-y.to_numpy()}).to_csv(OUT / 'b1_grouped_oof_predictions.csv', index=False)
print('\nMODEL SAVED SUCCESSFULLY')
print(f'Model: {OUT / "milkoptic_roi70_ridge.joblib"}')
print(f'Metadata: {OUT / "model_metadata.json"}')
print(f'B1 grouped-CV MAE: {metrics["B1_grouped_CV_MAE_percentage_points"]:.2f} percentage points')
print(f'B1 grouped-CV RMSE: {metrics["B1_grouped_CV_RMSE_percentage_points"]:.2f} percentage points')
print(f'B1 grouped-CV R2: {metrics["B1_grouped_CV_R2"]:.3f}')
print('Reminder: B2 exploratory MAE from V14 was 11.78 percentage points; fresh independent validation is still needed.')
