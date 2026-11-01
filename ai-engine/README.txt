# MilkOptic AI - ROI70 Prediction Package

## What this package does
Runs the saved ROI70 Ridge model on one milk image and returns an estimated water-dilution percentage.

It reproduces the V6 feature extraction sequence:
1. Resize image so the longest side is at most 1000 px (INTER_AREA).
2. Build the centered ellipse mask using 70% of the shorter image dimension.
3. Extract RGB, HSV, grayscale, 16-bin grayscale histogram, edge, Laplacian and ROI features.
4. Order the 55 values using `feature_names` from `model_metadata.json`.
5. Load and run the saved `.joblib` model.

## Files needed in your project
Keep the existing model files in:
`ai-engine/models/roi70_ridge/`
- `milkoptic_roi70_ridge.joblib`
- `model_metadata.json`

Copy `milkoptic_predict.py` into the `ai-engine` root.

## Install dependencies (inside your existing .venv)
```powershell
pip install opencv-python numpy joblib
```
If already installed, do not reinstall.

## Test one image
From `ai-engine`:
```powershell
python milkoptic_predict.py "D:\path\to\milk_photo.jpg"
```

Or:
```powershell
python milkoptic_predict.py "D:\path\to\milk_photo.jpg" --model-dir "models\roi70_ridge"
```

Successful output is JSON with `estimated_water_dilution_percent`.

## Important
- This is an experimental water-dilution estimate, not a lab test or food-safety certification.
- It does not detect detergent, soybean, or other adulterants.
- B1 grouped cross-validation is internal validation; B2 was exploratory. Fresh independent samples are still required.
- This script assumes the saved model metadata contains the `feature_names` array used during training.
