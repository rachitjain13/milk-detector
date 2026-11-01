import zipfile
import pandas as pd

zip_path = r"D:\A Spectral Dataset of both Liquid and Solid Milk with Impurities for some Milk Impurity Detection.zip"

with zipfile.ZipFile(zip_path) as z:
    for name in z.namelist():
        if not name.lower().endswith(".csv"):
            continue

        try:
            df = pd.read_csv(z.open(name))
            df.columns = df.columns.astype(str).str.strip()

            print("\n" + "=" * 65)
            print("FILE:", name.split("/")[-1])
            print("ROWS:", len(df))
            print("COLUMNS:", list(df.columns))

            for col in df.columns:
                if col.strip().lower() in ["label", "class", "status", "target"]:
                    print("LABEL COUNTS:")
                    print(df[col].astype(str).str.strip().value_counts().to_string())

            print("MISSING VALUES:", int(df.isna().sum().sum()))

        except Exception as e:
            print("ERROR:", name, str(e))
