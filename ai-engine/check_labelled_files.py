import zipfile
import pandas as pd

p = r"D:\A Spectral Dataset of both Liquid and Solid Milk with Impurities for some Milk Impurity Detection.zip"

with zipfile.ZipFile(p) as z:
    files = [n for n in z.namelist()
             if n.lower().endswith(".csv")
             and n.endswith("-labelled.csv")]

    for name in files:
        d = pd.read_csv(z.open(name))
        d.columns = d.columns.astype(str).str.strip()
        features = [f"Diode {i}" for i in range(1, 9)]

        print("\nFILE:", name.split("/")[-1])
        print("Rows:", len(d))
        print("Label:", d["Label"].value_counts().to_dict())
        print("Unique feature rows:", d[features].drop_duplicates().shape[0])
        print("Timestamp range:",
              d.iloc[0, 0], "to", d.iloc[-1, 0])
