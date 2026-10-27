
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folders
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# Original Transformer losses recorded during training
original_train = [
    11571.65, 6310.88, 2127.14, 1228.97,
    1048.70, 989.55, 942.18, 891.07,
    824.74, 728.60, 659.33, 593.28
]
original_val = [
    8450.09, 2685.59, 939.27, 659.27,
    638.29, 685.75, 713.10, 735.36,
    781.54, 813.60, 993.38, 1062.42
]

# Load regularized model's recorded history
history_path = RESULTS / "transformer_regularized_history.csv"
df = pd.read_csv(history_path)

print("Columns in history file:", df.columns.tolist())
print(df.head())

# Automatically identify loss columns
train_col = next(
    (c for c in df.columns if "train" in c.lower() and "mse" in c.lower()),
    None
)
val_col = next(
    (c for c in df.columns if ("val" in c.lower() or "validation" in c.lower())
     and "mse" in c.lower()),
    None
)

if train_col is None or val_col is None:
    raise ValueError(
        "Could not identify loss columns. Check the printed column names "
        "and update train_col / val_col."
    )

regular_train = df[train_col].tolist()
regular_val = df[val_col].tolist()

# Plot training losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, 13), original_train, marker="o",
         label="Original - Training MSE")
plt.plot(range(1, 13), original_val, marker="o",
         label="Original - Validation MSE")
plt.plot(range(1, len(regular_train) + 1), regular_train, marker="s",
         label="Regularized - Training MSE")
plt.plot(range(1, len(regular_val) + 1), regular_val, marker="s",
         label="Regularized - Validation MSE")

plt.axvline(5, linestyle="--", label="Original best epoch: 5")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Original vs Regularized Transformer: Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = PLOTS / "overfitting_comparison.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nGraph saved to: {output_path}")
