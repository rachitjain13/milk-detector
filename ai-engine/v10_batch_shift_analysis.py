"""
MilkOptic AI V10 - Batch / Zoom / Dilution Interaction Analysis

Inputs:
  v8_b1_grouped_oof_predictions.csv
  v8_b2_exploratory_predictions.csv

This script analyzes saved V8 predictions. It does not retrain or tune a model.
B1 grouped OOF is validation within B1. B2 remains exploratory, not independent
validation and must not be used to select/tune the model.

Run from ai-engine:
  python v10_batch_shift_analysis.py
Optional:
  python v10_batch_shift_analysis.py --input-dir analysis_output/milk_segmentation_v8_model_improvement
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL = "Ridge_scaled"
EXPERIMENT = "FULL_IMAGE"

def load_selected(path, prediction_col, batch_label):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    required = {"model", "experiment", "actual", "zoom", prediction_col, "file_name", "relative_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    df["experiment"] = df["experiment"].fillna("FULL_IMAGE")
    df = df[(df["model"] == MODEL) & (df["experiment"] == EXPERIMENT)].copy()
    df["batch"] = df["batch"].fillna(batch_label) if "batch" in df else batch_label
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["zoom"] = pd.to_numeric(df["zoom"], errors="coerce")
    df["prediction"] = pd.to_numeric(df[prediction_col], errors="coerce")
    df = df.dropna(subset=["actual", "zoom", "prediction"])
    df["error"] = df["prediction"] - df["actual"]
    df["absolute_error"] = df["error"].abs()
    df["batch"] = batch_label
    return df

def summarize(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        e = g["error"].to_numpy()
        row.update({
            "n": int(len(g)),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias_prediction_minus_actual": float(np.mean(e)),
            "mean_prediction": float(g["prediction"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)

def save_heatmap(table, value, title, out_path):
    pivot = table.pivot(index="actual", columns="zoom", values=value).sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    ax.set_xlabel("Zoom")
    ax.set_ylabel("Actual water dilution (%)")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label=value)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="analysis_output/milk_segmentation_v8_model_improvement")
    ap.add_argument("--output-dir", default="analysis_output/milk_segmentation_v10_batch_shift")
    args = ap.parse_args()
    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    b1 = load_selected(inp / "v8_b1_grouped_oof_predictions.csv", "prediction_oof", "B1")
    b2 = load_selected(inp / "v8_b2_exploratory_predictions.csv", "prediction", "B2")
    all_df = pd.concat([b1, b2], ignore_index=True)

    # Overall and zoom-only summaries
    overall = summarize(all_df, ["batch"])
    zoom = summarize(all_df, ["batch", "zoom"])
    dilution = summarize(all_df, ["batch", "actual"])
    interaction = summarize(all_df, ["batch", "actual", "zoom"])
    overall.to_csv(out / "v10_batch_overall_metrics.csv", index=False)
    zoom.to_csv(out / "v10_batch_zoom_metrics.csv", index=False)
    dilution.to_csv(out / "v10_batch_dilution_metrics.csv", index=False)
    interaction.to_csv(out / "v10_dilution_by_zoom_metrics.csv", index=False)

    # Within each batch, compare same dilution + zoom cells (descriptive only).
    compare = interaction.pivot_table(
        index=["actual", "zoom"], columns="batch",
        values=["n", "mae", "rmse", "bias_prediction_minus_actual", "mean_prediction"]
    )
    compare.to_csv(out / "v10_matched_dilution_zoom_batch_comparison.csv")

    # Duplicate and metadata checks. These checks do not inspect image pixels.
    audit_rows = []
    for name, df in [("B1", b1), ("B2", b2)]:
        for cols, label in [
            (["file_name"], "duplicate_file_name"),
            (["relative_path"], "duplicate_relative_path"),
        ]:
            audit_rows.append({
                "batch": name, "check": label,
                "duplicate_rows": int(df.duplicated(cols).sum()),
                "note": "Filename/path duplicate check only; not image-content duplicate detection."
            })
        audit_rows.append({
            "batch": name, "check": "unique_zoom_values",
            "duplicate_rows": 0,
            "note": ", ".join(map(str, sorted(df["zoom"].dropna().unique())))
        })
        audit_rows.append({
            "batch": name, "check": "rows_per_dilution_zoom_cell",
            "duplicate_rows": 0,
            "note": str(df.groupby(["actual", "zoom"]).size().to_dict())
        })
    pd.DataFrame(audit_rows).to_csv(out / "v10_metadata_duplicate_audit.csv", index=False)

    # Plots
    for batch, df in [("B1", b1), ("B2 exploratory", b2)]:
        tab = summarize(df, ["actual", "zoom"])
        save_heatmap(tab, "mae", f"{batch}: MAE by dilution and zoom",
                     out / f"v10_{batch.split()[0].lower()}_mae_heatmap.png")
        save_heatmap(tab, "bias_prediction_minus_actual",
                     f"{batch}: prediction bias by dilution and zoom",
                     out / f"v10_{batch.split()[0].lower()}_bias_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for batch, df in [("B1 grouped OOF", b1), ("B2 exploratory", b2)]:
        s = summarize(df, ["actual"])
        ax.plot(s["actual"], s["mae"], marker="o", label=batch)
    ax.set_xlabel("Actual water dilution (%)")
    ax.set_ylabel("MAE (percentage points)")
    ax.set_title("Concentration-wise MAE: B1 vs B2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "v10_b1_vs_b2_concentration_mae.png", dpi=160)
    plt.close(fig)

    # Key numerical facts in JSON, without claiming causality.
    def pack(df):
        e = df["error"].to_numpy()
        return {"n": int(len(df)), "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e**2))), "bias": float(np.mean(e))}
    summary = {
        "model": f"{EXPERIMENT} / {MODEL}",
        "B1_grouped_OOF": pack(b1),
        "B2_exploratory_only": pack(b2),
        "B1_zoom": summarize(b1, ["zoom"]).to_dict(orient="records"),
        "B2_zoom_exploratory": summarize(b2, ["zoom"]).to_dict(orient="records"),
        "limitations": [
            "B2 is exploratory, not independent validation, and must not be used for model selection or tuning.",
            "B1 grouped OOF is validation within B1 and does not establish real-world performance.",
            "The prediction CSVs do not contain original image pixel features such as brightness, contrast, sharpness or color statistics.",
            "Zoom associations are descriptive and do not prove zoom causes prediction errors.",
            "Filename/path checks do not detect visually identical images under different filenames."
        ]
    }
    (out / "v10_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "README_V10.txt").write_text(
        "V10 completed: batch, dilution, zoom interaction analysis.\\n"
        "Selected model: FULL_IMAGE / Ridge_scaled (selection from B1 only).\\n"
        "B2 results are exploratory only; do not tune/select using B2.\\n"
        "No model was retrained in V10.\\n", encoding="utf-8"
    )

    print("\\nV10 ANALYSIS COMPLETE")
    print("Selected model:", EXPERIMENT, "/", MODEL)
    print("\\nOverall:")
    print(overall.to_string(index=False))
    print("\\nBy batch and zoom:")
    print(zoom.to_string(index=False))
    print("\\nBy batch, dilution and zoom:")
    print(interaction.to_string(index=False))
    print("\\nOutputs saved to:", out.resolve())
    print("Reminder: B2 is exploratory only.")

if __name__ == "__main__":
    main()
