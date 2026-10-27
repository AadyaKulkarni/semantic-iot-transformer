
from pathlib import Path
import pandas as pd

# Location of your NASA C-MAPSS dataset
DATA_DIR = Path(
    r"C:\Users\adku1\Downloads\archive (1)\CMaps"
)

# Check that the folder exists
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

print("Dataset folder found!")

# Display available files
print("\nDataset files:")
for file in sorted(DATA_DIR.iterdir()):
    if file.is_file():
        print(file.name)

# Load FD001 training data
train_file = DATA_DIR / "train_FD001.txt"

if not train_file.exists():
    raise FileNotFoundError(f"Missing file: {train_file}")

columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

train = pd.read_csv(
    train_file,
    sep=r"\s+",
    header=None,
    names=columns
)

print("\nDataset shape:", train.shape)
print("Number of engines:", train["engine_id"].nunique())
print("\nFirst five rows:")
print(train.head())

print("\nMissing values:", train.isna().sum().sum())
