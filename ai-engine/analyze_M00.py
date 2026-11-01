import pandas as pd

df = pd.read_csv("analysis_output/M00_phone_features.csv")

df["zoom"] = df["relative_path"].str.replace("\\", "/", regex=False).str.split("/").str[0]

print("Camera-wise image count:")
print(df["zoom"].value_counts())

print("\nCamera-wise brightness:")
print(df.groupby("zoom")["image_mean_brightness_gray"].agg(["mean", "std"]))

print("\nCamera-wise blur:")
print(df.groupby("zoom")["image_blur_laplacian_var"].agg(["mean", "std"]))