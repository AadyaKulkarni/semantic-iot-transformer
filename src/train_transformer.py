
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent.parent
SEQ_DIR = PROJECT_DIR / "data" / "sequences"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 128
EPOCHS = 12
LEARNING_RATE = 0.001

# Load prepared sequences
train_data = np.load(SEQ_DIR / "train_sequences.npz")
val_data = np.load(SEQ_DIR / "validation_sequences.npz")

X_train = torch.tensor(train_data["X"], dtype=torch.float32)
y_train = torch.tensor(train_data["y"], dtype=torch.float32)
X_val = torch.tensor(val_data["X"], dtype=torch.float32)
y_val = torch.tensor(val_data["y"], dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)


class RULTransformer(nn.Module):
    def __init__(self, input_dim=21, d_model=32, nhead=4, num_layers=2):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        # Learnable positional embeddings for 30 time steps
        self.position_embedding = nn.Parameter(
            torch.randn(1, 30, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position_embedding
        x = self.transformer(x)

        # Predict from the final time step
        x = x[:, -1, :]
        return self.regressor(x).squeeze(-1)


model = RULTransformer().to(DEVICE)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
criterion = nn.MSELoss()

best_val_loss = float("inf")
best_state = None

print("Training Transformer on:", DEVICE)
print("Training sequences:", len(X_train))
print("Validation sequences:", len(X_val))

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        predictions = model(xb)
        loss = criterion(predictions, yb)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * len(xb)

    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            predictions = model(xb)
            loss = criterion(predictions, yb)
            val_loss_sum += loss.item() * len(xb)

    train_loss = train_loss_sum / len(X_train)
    val_loss = val_loss_sum / len(X_val)

    print(
        f"Epoch {epoch + 1:02}/{EPOCHS} | "
        f"Train MSE: {train_loss:.2f} | "
        f"Validation MSE: {val_loss:.2f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

# Restore best validation checkpoint
model.load_state_dict(best_state)

# Final validation predictions
model.eval()
with torch.no_grad():
    predictions = model(X_val).numpy()

actual = y_val.numpy()

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
r2 = r2_score(actual, predictions)

print("\nTransformer Validation Results")
print("--------------------------------")
print(f"MAE:  {mae:.2f} cycles")
print(f"RMSE: {rmse:.2f} cycles")
print(f"R²:   {r2:.4f}")

# Save model and results
torch.save(model.state_dict(), MODEL_DIR / "rul_transformer.pt")

pd.DataFrame({
    "actual_RUL": actual,
    "predicted_RUL": predictions
}).to_csv(RESULTS_DIR / "transformer_predictions.csv", index=False)

pd.DataFrame([{
    "model": "Transformer",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "best_validation_MSE": best_val_loss
}]).to_csv(RESULTS_DIR / "transformer_metrics.csv", index=False)

print("\nSaved model to:", MODEL_DIR / "rul_transformer.pt")
print("Saved results to:", RESULTS_DIR)
