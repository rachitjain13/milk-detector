"""MilkOptic AI V8 - grouped model comparison and exploratory cross-batch test.

Run from ai-engine:
    python v8_model_improvement.py

Inputs (existing V6/V7 files; never overwritten):
    analysis_output/camera_experiments/unified_all_batches.csv
    analysis_output/milk_segmentation_v6_roi/v6_roi_features.csv

Outputs:
    analysis_output/milk_segmentation_v8_model_improvement/

Scientific guardrails:
- Model selection uses B1 grouped CV only (held-out concentration groups).
- B2 is evaluated once as an exploratory cross-batch check, not used to select models.
- B2 has been examined in V7, so it is NOT an untouched independent test set.
- This estimates water-dilution percentage only; it does not validate other adulterants.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parent
UNIFIED = ROOT / "analysis_output" / "camera_experiments" / "unified_all_batches.csv"
ROI_CSV = ROOT / "analysis_output" / "milk_segmentation_v6_roi" / "v6_roi_features.csv"
OUT = ROOT / "analysis_output" / "milk_segmentation_v8_model_improvement"
KEYS = ["batch", "concentration", "dilution_percent", "zoom", "file_name", "relative_path"]
META = set(KEYS) | {"roi_size", "format", "camera_mode", "image_width", "image_height", "width", "height", "row_index"}
SEED = 42


def norm_numeric_text(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.map(lambda v: "<NA>" if pd.isna(v) else f"{v:.8g}")


def normalize_keys(df, label, unique_columns=None):
    missing = [c for c in KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing key columns: {missing}")
    d = df.copy()
    for c in KEYS:
        if c == "dilution_percent":
            d[c] = norm_numeric_text(d[c])
        else:
            d[c] = d[c].astype("string").str.strip()
    if d[KEYS].isna().any().any():
        raise ValueError(f"{label} has missing image keys")
    unique_columns = unique_columns or []
    check_columns = KEYS + unique_columns
    if d.duplicated(check_columns).any():
        raise ValueError(f"{label} has duplicate keys: {check_columns}")
    return d


def numeric_features(df, excluded=META):
    cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    # Remove constant/all-empty columns; the imputer handles remaining missing values.
    cols = [c for c in cols if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    if not cols:
        raise ValueError("No usable numeric feature columns found")
    return cols


def model_zoo():
    return {
        "ExtraTrees_leaf2": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2, max_features=1.0, random_state=SEED, n_jobs=-1))]),
        "ExtraTrees_leaf5": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5, max_features=1.0, random_state=SEED, n_jobs=-1))]),
        "RandomForest_leaf2": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.8, random_state=SEED, n_jobs=-1))]),
        "HistGradientBoosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, l2_regularization=2.0, random_state=SEED))]),
        "Ridge_scaled": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
    }


def metrics(y, p):
    return {"mae": mean_absolute_error(y, p), "rmse": np.sqrt(mean_squared_error(y, p)), "r2": r2_score(y, p)}


def evaluate_grouped_cv(frame, feature_cols, model_name, estimator):
    b1 = frame[frame.batch.astype(str).str.upper().eq("B1")].copy().reset_index(drop=True)
    y = pd.to_numeric(b1.dilution_percent, errors="raise").to_numpy(float)
    groups = b1.concentration.astype(str).to_numpy()
    X = b1[feature_cols].apply(pd.to_numeric, errors="coerce")
    logo = LeaveOneGroupOut()
    pred = np.full(len(b1), np.nan)
    fold_rows = []
    for fold, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        m = clone(estimator)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        pred[te] = p
        row = metrics(y[te], p)
        fold_rows.append({"model": model_name, "held_out_concentration": groups[te][0], "n_test": len(te), **row})
    ok = np.isfinite(pred)
    overall = metrics(y[ok], pred[ok])
    oof = b1[KEYS].copy()
    oof["model"] = model_name
    oof["actual"] = y
    oof["prediction_oof"] = pred
    oof["absolute_error"] = np.abs(y - pred)
    oof["signed_error"] = pred - y
    return overall, oof, fold_rows


def evaluate_b2(frame, feature_cols, model_name, estimator):
    b1 = frame[frame.batch.astype(str).str.upper().eq("B1")].copy()
    b2 = frame[frame.batch.astype(str).str.upper().eq("B2")].copy()
    if b1.empty or b2.empty:
        raise ValueError("Both B1 and B2 rows are required")
    X1 = b1[feature_cols].apply(pd.to_numeric, errors="coerce")
    X2 = b2[feature_cols].apply(pd.to_numeric, errors="coerce")
    y1 = pd.to_numeric(b1.dilution_percent, errors="raise").to_numpy(float)
    y2 = pd.to_numeric(b2.dilution_percent, errors="raise").to_numpy(float)
    m = clone(estimator)
    m.fit(X1, y1)
    p = m.predict(X2)
    out = b2[KEYS].copy()
    out["model"] = model_name
    out["actual"] = y2
    out["prediction"] = p
    out["absolute_error"] = np.abs(y2 - p)
    out["signed_error"] = p - y2
    return metrics(y2, p), out


def main():
    for path in (UNIFIED, ROI_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    base = normalize_keys(pd.read_csv(UNIFIED), "unified dataset")
    roi = normalize_keys(pd.read_csv(ROI_CSV), "V6 ROI features", unique_columns=["roi_size"])
    if "roi_size" not in roi.columns:
        raise ValueError("V6 ROI CSV must contain roi_size")
    base["batch"] = base.batch.astype(str).str.upper()
    roi["batch"] = roi.batch.astype(str).str.upper()

    all_metrics, all_oof, all_b2, all_folds = [], [], [], []
    zoo = model_zoo()

    # Full-image baseline uses numeric feature columns from the unified manifest.
    full_cols = numeric_features(base)
    print(f"Unified rows={len(base)} | full-image features={len(full_cols)}")
    for name, est in zoo.items():
        print(f"CV full image: {name}")
        cvm, oof, folds = evaluate_grouped_cv(base, full_cols, name, est)
        b2m, b2pred = evaluate_b2(base, full_cols, name, est)
        all_metrics += [{"experiment": "FULL_IMAGE", "model": name, "validation": "B1_LOCO", "n_features": len(full_cols), **cvm}, {"experiment": "FULL_IMAGE", "model": name, "validation": "B1_to_B2_exploratory", "n_features": len(full_cols), **b2m}]
        all_oof.append(oof); all_b2.append(b2pred); all_folds.extend(folds)

    # ROI features: test each geometry; model selection still based on B1 grouped CV.
    for size in (60, 70, 80):
        part = roi[pd.to_numeric(roi.roi_size, errors="coerce").eq(size)].copy()
        # ROI rows already contain the complete KEYS, including batch.
        # Do not merge against the manifest here: KEYS already includes batch,
        # and the previous merge accidentally removed that column from the RHS.
        if not part.empty:
            part["batch"] = part["batch"].astype(str).str.upper()
        if part.empty:
            raise ValueError(f"No matched rows for ROI {size}%")
        cols = numeric_features(part)
        print(f"ROI {size}% rows={len(part)} | features={len(cols)}")
        for name, est in zoo.items():
            print(f"CV ROI {size}%: {name}")
            cvm, oof, folds = evaluate_grouped_cv(part, cols, name, est)
            b2m, b2pred = evaluate_b2(part, cols, name, est)
            exp = f"CENTRAL_ELLIPSE_{size}PCT"
            all_metrics += [{"experiment": exp, "model": name, "validation": "B1_LOCO", "n_features": len(cols), **cvm}, {"experiment": exp, "model": name, "validation": "B1_to_B2_exploratory", "n_features": len(cols), **b2m}]
            oof["experiment"] = exp; b2pred["experiment"] = exp
            all_oof.append(oof); all_b2.append(b2pred); all_folds.extend(folds)

    mdf = pd.DataFrame(all_metrics)
    # Choose a model per experiment using B1 LOCO MAE only; B2 does not influence selection.
    selection = (mdf[mdf.validation.eq("B1_LOCO")].sort_values(["experiment", "mae", "rmse"]).groupby("experiment", as_index=False).first())
    selection.to_csv(OUT / "v8_model_selection_b1_only.csv", index=False)
    mdf.to_csv(OUT / "v8_all_metrics.csv", index=False)
    pd.concat(all_oof, ignore_index=True).to_csv(OUT / "v8_b1_grouped_oof_predictions.csv", index=False)
    pd.concat(all_b2, ignore_index=True).to_csv(OUT / "v8_b2_exploratory_predictions.csv", index=False)
    pd.DataFrame(all_folds).to_csv(OUT / "v8_b1_fold_metrics.csv", index=False)

    b2 = pd.concat(all_b2, ignore_index=True)
    conc = b2.groupby(["experiment", "model", "actual"], as_index=False).agg(n=("absolute_error", "size"), mae=("absolute_error", "mean"), mean_prediction=("prediction", "mean"), mean_signed_error=("signed_error", "mean"))
    conc.to_csv(OUT / "v8_b2_concentration_errors.csv", index=False)

    print("\nV8 BEST MODELS (selected using B1 grouped CV only)")
    print(selection[["experiment", "model", "n_features", "mae", "rmse", "r2"]].to_string(index=False))
    print(f"\nSaved outputs to: {OUT}")
    print("NOTE: B2 results are exploratory; do not use B2 to choose the final model.")

if __name__ == "__main__":
    main()
