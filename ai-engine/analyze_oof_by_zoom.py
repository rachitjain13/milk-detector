
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent

INPUT = (
    BASE / "analysis_output" / "camera_experiments"
    / "oof_analysis" / "combined_oof_predictions.csv"
)

OUTPUT = INPUT.parent / "combined_oof_by_zoom.csv"

df = pd.read_csv(INPUT)

# Zoom-wise performance
zoom_summary = (
    df.groupby(["concentration", "zoom"])
    .agg(
        images=("absolute_error", "size"),
        actual=("dilution_percent", "first"),
        mean_predicted=("predicted_percent", "mean"),
        MAE=("absolute_error", "mean"),
        RMSE=(
            "absolute_error",
            lambda x: np.sqrt(np.mean(x ** 2))
        )
    )
    .reset_index()
)

print("\nZOOM-WISE PERFORMANCE")
print(zoom_summary.to_string(index=False))

# Camera-mode breakdown, if metadata exists
if "camera_mode" in df.columns:
    mode_summary = (
        df.groupby(["concentration", "camera_mode"])
        .agg(
            images=("absolute_error", "size"),
            MAE=("absolute_error", "mean"),
            mean_predicted=("predicted_percent", "mean")
        )
        .reset_index()
    )

    print("\nCAMERA-MODE PERFORMANCE")
    print(mode_summary.to_string(index=False))

zoom_summary.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
