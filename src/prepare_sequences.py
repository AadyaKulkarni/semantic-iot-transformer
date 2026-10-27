
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "data" / "sequences"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 30
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

def create_sequences(df):
    X, y = [], []

    # Keep each engine's sequence separate
    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")

        sensor_values = group[SENSOR_COLS].to_numpy(dtype=np.float32)
        rul_values = group["RUL"].to_numpy(dtype=np.float32)

        # Use consecutive windows of 30 cycles
        for end in range(SEQ_LEN - 1, len(group)):
            start = end - SEQ_LEN + 1

            X.append(sensor_values[start:end + 1])
            y.append(rul_values[end])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

X_train, y_train = create_sequences(train_df)
X_val, y_val = create_sequences(val_df)

np.savez_compressed(
    OUTPUT_DIR / "train_sequences.npz",
    X=X_train,
    y=y_train
)

np.savez_compressed(
    OUTPUT_DIR / "validation_sequences.npz",
    X=X_val,
    y=y_val
)

print("Sequence preparation complete!")
print("Training X shape:", X_train.shape)
print("Training y shape:", y_train.shape)
print("Validation X shape:", X_val.shape)
print("Validation y shape:", y_val.shape)
print("Sequence length:", SEQ_LEN)
print("Sensor features:", len(SENSOR_COLS))
print("Saved to:", OUTPUT_DIR)
