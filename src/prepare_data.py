
from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\adku1\Downloads\archive (1)\CMaps"
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load training data
train = pd.read_csv(
    DATA_DIR / "train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# Find each engine's last recorded cycle
last_cycle = train.groupby("engine_id")["cycle"].max()

# Calculate RUL for every row
train["RUL"] = (
    train["engine_id"].map(last_cycle) - train["cycle"]
)

# Save the prepared dataset
output_file = OUTPUT_DIR / "train_FD001_RUL.csv"
train.to_csv(output_file, index=False)

print("RUL calculation complete!")
print("\nDataset shape:", train.shape)
print("\nFirst 10 rows:")
print(train[["engine_id", "cycle", "RUL"]].head(10))

print("\nRUL statistics:")
print(train["RUL"].describe())

print("\nSaved to:", output_file)
