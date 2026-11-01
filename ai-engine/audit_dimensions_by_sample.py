import pandas as pd

df = pd.read_csv("analysis_output/image_format_audit.csv")

summary = (
    df.groupby(["batch", "concentration", "zoom", "format", "width", "height"])
    .size()
    .reset_index(name="image_count")
)

print("\n===== DIMENSIONS BY SAMPLE =====")
print(summary.to_string(index=False))

print("\n===== 8992 PIXEL IMAGES =====")
unusual = df[(df["width"] == 8992) | (df["height"] == 8992)]

if len(unusual):
    print(
        unusual.groupby(
            ["batch", "concentration", "zoom", "format"]
        ).size().to_string()
    )
else:
    print("No unusual images found.")

summary.to_csv(
    "analysis_output/dimensions_by_sample.csv",
    index=False
)

print("\nSaved: analysis_output/dimensions_by_sample.csv")