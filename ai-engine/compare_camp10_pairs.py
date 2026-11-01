
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).resolve().parent

INPUT = HERE / "analysis_output" / "camp10_texture_statistics.csv"
OUTPUT = HERE / "analysis_output" / "camp10_pair_comparison.png"
CSV_OUTPUT = HERE / "analysis_output" / "camp10_pair_summary.csv"

df = pd.read_csv(INPUT)

pairs = [
    ("R", "RAD"),
    ("S1", "SAD1"),
    ("S2", "SAD2"),
    ("S3", "SAD3"),
    ("S4", "SAD4"),
    ("S5", "SAD5")
]

features = ["mean_brightness", "entropy", "sharpness", "lbp_mean"]

summary = []

for normal, adulterated in pairs:
    a = df[df["category"] == normal]
    b = df[df["category"] == adulterated]

    for feature in features:
        summary.append({
            "pair": f"{normal} vs {adulterated}",
            "feature": feature,
            "unadulterated_mean": a[feature].mean(),
            "adulterated_mean": b[feature].mean(),
            "difference": b[feature].mean() - a[feature].mean()
        })

result = pd.DataFrame(summary)
result.to_csv(CSV_OUTPUT, index=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, feature in zip(axes.flat, features):
    sns.boxplot(
        data=df,
        x="category",
        y=feature,
        ax=ax
    )
    ax.set_title(feature.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)

plt.suptitle("CAMP10 Category-wise Feature Distributions")
plt.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=150)

print("\nPAIR-WISE COMPARISON:")
print(result.round(3).to_string(index=False))

print("\nSaved:", CSV_OUTPUT)
print("Saved:", OUTPUT)

plt.show()
