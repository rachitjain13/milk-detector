"""
MilkOptic AI - V9 Reliability & Error Analysis
Uses V8 CSV outputs only. It does NOT retrain models or prove real-world validity.

Expected input files in --input-dir:
  v8_all_metrics.csv
  v8_b1_fold_metrics.csv
  v8_b1_grouped_oof_predictions.csv
  v8_b2_concentration_errors.csv
  v8_b2_exploratory_predictions.csv
  v8_model_selection_b1_only.csv

Run:
  python v9_reliability_analysis.py
or:
  python v9_reliability_analysis.py --input-dir analysis_output/milk_segmentation_v8_model_improvement
"""

from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILES = [
    "v8_all_metrics.csv",
    "v8_b1_fold_metrics.csv",
    "v8_b1_grouped_oof_predictions.csv",
    "v8_b2_concentration_errors.csv",
    "v8_b2_exploratory_predictions.csv",
    "v8_model_selection_b1_only.csv",
]

def read_csv(folder, name):
    p = folder / name
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}\nCopy all six V8 CSV files into the V8 output folder, "
            "or pass the folder containing them with --input-dir."
        )
    return pd.read_csv(p)

def metrics_frame(df, pred_col):
    x = df.copy()
    x["error"] = x[pred_col] - x["actual"]
    x["absolute_error_calc"] = x["error"].abs()
    rows = []
    for level, g in x.groupby("actual", dropna=False):
        rows.append({
            "actual_dilution": level,
            "n": len(g),
            "mae": g["absolute_error_calc"].mean(),
            "rmse": np.sqrt(np.mean(g["error"] ** 2)),
            "mean_prediction": g[pred_col].mean(),
            "mean_signed_error_prediction_minus_actual": g["error"].mean(),
            "prediction_std": g[pred_col].std(),
        })
    return pd.DataFrame(rows).sort_values("actual_dilution")

def overall_metrics(df, pred_col):
    err = df[pred_col] - df["actual"]
    return {
        "n": len(df),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias_prediction_minus_actual": float(np.mean(err)),
    }

def save_scatter(df, pred_col, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["actual"], df[pred_col], alpha=0.22, s=12)
    lo = min(float(df["actual"].min()), float(df[pred_col].min()))
    hi = max(float(df["actual"].max()), float(df[pred_col].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual water dilution (%)")
    ax.set_ylabel("Predicted water dilution (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="analysis_output/milk_segmentation_v8_model_improvement")
    ap.add_argument("--output-dir", default="analysis_output/milk_segmentation_v9_reliability")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = read_csv(input_dir, FILES[0])
    fold_metrics = read_csv(input_dir, FILES[1])
    b1_oof = read_csv(input_dir, FILES[2])
    b2_errors = read_csv(input_dir, FILES[3])
    b2_pred = read_csv(input_dir, FILES[4])
    selected = read_csv(input_dir, FILES[5])

    # V8 saved the FULL_IMAGE experiment as blank in prediction CSVs.
    # In these files, the blank experiment corresponds to FULL_IMAGE.
    for df in (b1_oof, b2_pred):
        df["experiment"] = df["experiment"].fillna("FULL_IMAGE")

    # Recalculate errors from predictions instead of trusting precomputed columns.
    b1_oof["prediction_oof"] = pd.to_numeric(b1_oof["prediction_oof"], errors="coerce")
    b1_oof["actual"] = pd.to_numeric(b1_oof["actual"], errors="coerce")
    b2_pred["prediction"] = pd.to_numeric(b2_pred["prediction"], errors="coerce")
    b2_pred["actual"] = pd.to_numeric(b2_pred["actual"], errors="coerce")
    b1_oof = b1_oof.dropna(subset=["actual", "prediction_oof"])
    b2_pred = b2_pred.dropna(subset=["actual", "prediction"])

    # Focus on the model selected using B1 grouped CV.
    chosen = selected.sort_values("mae").iloc[0]
    best_exp = str(chosen["experiment"])
    best_model = str(chosen["model"])

    best_b1 = b1_oof[
        (b1_oof["experiment"] == best_exp) & (b1_oof["model"] == best_model)
    ].copy()
    if best_b1.empty:
        raise ValueError(f"No B1 OOF predictions found for {best_exp} / {best_model}")

    best_b2 = b2_pred[
        (b2_pred["experiment"] == best_exp) & (b2_pred["model"] == best_model)
    ].copy()

    # B1 concentration-wise performance and bias.
    b1_by_level = metrics_frame(best_b1, "prediction_oof")
    b1_by_level.to_csv(output_dir / "v9_b1_concentration_analysis.csv", index=False)

    # B1 zoom analysis is diagnostic only; zoom may be a confounder.
    if "zoom" in best_b1.columns:
        zoom_rows = []
        for zoom, g in best_b1.groupby("zoom", dropna=False):
            m = overall_metrics(g, "prediction_oof")
            zoom_rows.append({"zoom": zoom, **m})
        pd.DataFrame(zoom_rows).to_csv(output_dir / "v9_b1_zoom_analysis.csv", index=False)

    # Fold stability limitation: the V8 fold CSV has no experiment/feature-set column.
    # Therefore rows for the same model can belong to multiple feature sets and cannot
    # be assigned safely to the selected FULL_IMAGE experiment.
    chosen_folds = fold_metrics[fold_metrics["model"] == best_model].copy()
    chosen_folds["interpretation_note"] = (
        "Model-only fold rows; V8 file lacks experiment column, so these cannot be "
        "attributed specifically to the selected feature set."
    )
    chosen_folds.to_csv(output_dir / "v9_model_fold_metrics_unassigned_feature_set.csv", index=False)

    # B2 is explicitly exploratory, NOT a model-selection or independent test set.
    if not best_b2.empty:
        b2_by_level = metrics_frame(best_b2, "prediction")
        b2_by_level.to_csv(output_dir / "v9_b2_exploratory_concentration_analysis.csv", index=False)
        pd.DataFrame([overall_metrics(best_b2, "prediction")]).to_csv(
            output_dir / "v9_b2_exploratory_overall_metrics.csv", index=False
        )

    # Dataset metadata / leakage warning checks using prediction manifests.
    audit = []
    for label, df in [("B1_OOF", best_b1), ("B2_EXPLORATORY", best_b2)]:
        if df.empty:
            continue
        for col in ["batch", "concentration", "dilution_percent", "zoom"]:
            if col in df.columns:
                audit.append({
                    "dataset": label,
                    "check": f"unique_{col}",
                    "value": str(df[col].nunique(dropna=False)),
                    "interpretation": "Metadata variation; inspect whether this could encode collection conditions."
                })
        for cols, name in [
            (["batch", "relative_path"], "duplicate_batch_relative_path"),
            (["batch", "file_name"], "duplicate_batch_file_name"),
        ]:
            if all(c in df.columns for c in cols):
                audit.append({
                    "dataset": label,
                    "check": name,
                    "value": str(int(df.duplicated(cols).sum())),
                    "interpretation": "Duplicate key count in this prediction table; not a complete image-content leakage test."
                })
    pd.DataFrame(audit).to_csv(output_dir / "v9_metadata_audit.csv", index=False)

    # Save plots.
    save_scatter(
        best_b1, "prediction_oof",
        f"B1 grouped OOF: {best_exp} / {best_model}",
        output_dir / "v9_b1_actual_vs_predicted.png"
    )
    if not best_b2.empty:
        save_scatter(
            best_b2, "prediction",
            f"B2 exploratory only: {best_exp} / {best_model}",
            output_dir / "v9_b2_actual_vs_predicted_exploratory.png"
        )

    # Concentration-wise B1 MAE and mean prediction plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(b1_by_level["actual_dilution"].astype(str), b1_by_level["mae"])
    ax.set_xlabel("Actual water dilution (%)")
    ax.set_ylabel("MAE (percentage points)")
    ax.set_title("B1 grouped OOF error by dilution level")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "v9_b1_mae_by_concentration.png", dpi=160)
    plt.close(fig)

    # Machine-readable summary and run notes.
    summary = {
        "selected_experiment_from_B1_only": best_exp,
        "selected_model_from_B1_only": best_model,
        "b1_grouped_oof": overall_metrics(best_b1, "prediction_oof"),
        "b1_prediction_count": int(len(best_b1)),
        "b2_exploratory_only": overall_metrics(best_b2, "prediction") if not best_b2.empty else None,
        "b2_prediction_count": int(len(best_b2)),
        "important_limitations": [
            "B2 is exploratory and must not be used to select or tune the model.",
            "B1 grouped OOF predictions are validation within one collection batch, not independent real-world validation.",
            "Fold R2 is not meaningful when each held-out fold contains only one target concentration; interpret MAE/RMSE instead.",
            "V8 fold-metrics CSV lacks the experiment/feature-set column, so fold rows cannot be attributed to the selected feature set.",
            "The uploaded V8 CSVs contain predictions and metadata, not the fitted Ridge coefficients or original feature matrix.",
            "This analysis cannot establish causality, laboratory accuracy, or detect adulterants other than the studied water dilution."
        ]
    }
    pd.Series(summary).to_json(output_dir / "v9_summary.json", indent=2)

    with open(output_dir / "README_V9.txt", "w", encoding="utf-8") as f:
        f.write(
            "V9 Reliability & Error Analysis completed.\\n"
            f"Selected using B1 only: {best_exp} / {best_model}\\n"
            f"B1 grouped OOF metrics: {summary['b1_grouped_oof']}\\n"
            "B2 is exploratory only. Do not use it for model selection.\\n"
            "This script analyzes existing V8 outputs; it does not retrain the model.\\n"
        )

    print("\\nV9 ANALYSIS COMPLETE")
    print(f"Selected using B1 only: {best_exp} / {best_model}")
    print("B1 grouped OOF:", summary["b1_grouped_oof"])
    if summary["b2_exploratory_only"] is not None:
        print("B2 exploratory only:", summary["b2_exploratory_only"])
    print(f"Outputs saved to: {output_dir.resolve()}")
    print("\\nImportant: B2 is exploratory, not an independent test set.")

if __name__ == "__main__":
    main()
