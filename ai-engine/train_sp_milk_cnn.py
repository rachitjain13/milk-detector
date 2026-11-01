import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras import layers, models, callbacks

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "models" / "sp_milk_cnn"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = OUT.parent / "sp_milk_hog_svm" / "image_manifest_with_splits.csv"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)
required = {"path", "label", "class_name", "split", "campaign"}
if not required.issubset(df.columns):
    raise ValueError(f"Manifest needs columns: {sorted(required)}")

print("TensorFlow:", tf.__version__)
print("CPU training:", not bool(tf.config.list_physical_devices("GPU")))
print("\nImages per split:")
print(df.groupby(["split", "class_name"]).size())

def load_images(part):
    X, y = [], []
    for i, row in enumerate(part.itertuples(index=False), 1):
        img = tf.keras.utils.load_img(
            row.path, color_mode="grayscale", target_size=IMG_SIZE
        )
        arr = tf.keras.utils.img_to_array(img) / 255.0
        X.append(arr)
        y.append(int(row.label))
        if i % 500 == 0:
            print(f"  Loaded {i}/{len(part)}")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "validation"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print("\nLoading training images...")
X_train, y_train = load_images(train_df)
print("\nLoading validation images...")
X_val, y_val = load_images(val_df)
print("\nLoading test images...")
X_test, y_test = load_images(test_df)

tf.keras.utils.set_random_seed(SEED)

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 1)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

cb = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    ),
    callbacks.ModelCheckpoint(
        str(OUT / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    verbose=1,
)

model = tf.keras.models.load_model(OUT / "best_model.keras")
prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).ravel()
pred = (prob >= 0.5).astype(np.int32)

metrics = {
    "test_accuracy": float(accuracy_score(y_test, pred)),
    "labels_order": ["adulterated", "unadulterated"],
    "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    "classification_report": classification_report(
        y_test, pred, labels=[0, 1],
        target_names=["adulterated", "unadulterated"],
        output_dict=True, zero_division=0,
    ),
    "test_images": int(len(y_test)),
    "image_size": list(IMG_SIZE),
    "train_campaigns": sorted(train_df["campaign"].unique().tolist()),
    "validation_campaigns": sorted(val_df["campaign"].unique().tolist()),
    "test_campaigns": sorted(test_df["campaign"].unique().tolist()),
}

with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nTEST ACCURACY:", round(metrics["test_accuracy"] * 100, 2), "%")
print("Confusion matrix [true rows, predicted columns]:")
print(np.array(metrics["confusion_matrix"]))
print(classification_report(
    y_test, pred, labels=[0, 1],
    target_names=["adulterated", "unadulterated"],
    zero_division=0,
))
print("\nSaved model:", OUT / "best_model.keras")
print("Saved metrics:", OUT / "metrics.json")
