
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load prepared data
df = pd.read_csv(DATA_DIR / "train_FD001_RUL.csv")

# Sensor features only
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

# Split by engine ID to avoid data leakage
engine_ids = df["engine_id"].unique()
train_ids, val_ids = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

train_df = df[df["engine_id"].isin(train_ids)].copy()
val_df = df[df["engine_id"].isin(val_ids)].copy()

# Fit scaler on training engines only
scaler = StandardScaler()
train_df.loc[:, sensor_cols] = scaler.fit_transform(
    train_df[sensor_cols]
)
val_df.loc[:, sensor_cols] = scaler.transform(
    val_df[sensor_cols]
)

# Save processed datasets and scaler
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

print("Preprocessing complete!")
print("Training engines:", len(train_ids))
print("Validation engines:", len(val_ids))
print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Sensor features:", len(sensor_cols))
print("Training RUL range:",
      train_df["RUL"].min(), "to", train_df["RUL"].max())
print("Saved to:", OUTPUT_DIR)
