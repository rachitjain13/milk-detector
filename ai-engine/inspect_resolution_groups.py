import pandas as pd

df = pd.read_csv("analysis_output/image_format_audit.csv")

print("\n===== RESOLUTION DISTRIBUTION =====")
print(
    df.groupby(
        ["batch", "concentration", "zoom", "width", "height"]
    ).size().to_string()
)

print("\n===== 8992 RESOLUTION IMAGES =====")
high = df[
    (df["width"] == 8992) &
    (df["height"] == 8992)
]

print(high[
    ["batch", "concentration", "zoom", "format", "file_size_kb"]
].to_string(index=False))

print("\nTotal high-resolution images:", len(high))