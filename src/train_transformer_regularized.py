
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
MAX_EPOCHS = 6
PATIENCE = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001

# Load exactly the same sequences as the original experiment
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
        self.position_embedding = nn.Parameter(
            torch.randn(1, 30, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.2,  # Increased dropout
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
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position_embedding
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.regressor(x).squeeze(-1)


model = RULTransformer().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
criterion = nn.MSELoss()

best_val_loss = float("inf")
best_state = None
best_epoch = 0
epochs_without_improvement = 0
history = []

print("Regularized Transformer training on:", DEVICE)
print("Dropout:", 0.2)
print("Weight decay:", WEIGHT_DECAY)
print("Early stopping patience:", PATIENCE)

for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss_sum = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        predictions = model(xb)
        loss = criterion(predictions, yb)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * len(xb)

    train_loss = train_loss_sum / len(X_train)

    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            predictions = model(xb)
            loss = criterion(predictions, yb)
            val_loss_sum += loss.item() * len(xb)

    val_loss = val_loss_sum / len(X_val)

    improved = val_loss < best_val_loss

    history.append({
        "epoch": epoch + 1,
        "train_mse": train_loss,
        "validation_mse": val_loss,
        "improved": improved
    })

    print(
        f"Epoch {epoch + 1:02}/{MAX_EPOCHS} | "
        f"Train MSE: {train_loss:.2f} | "
        f"Validation MSE: {val_loss:.2f}"
    )

    if improved:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

# Restore best checkpoint
model.load_state_dict(best_state)

# Evaluate the best checkpoint
model.eval()
predictions_list = []

with torch.no_grad():
    for xb, _ in val_loader:
        predictions_list.extend(model(xb).numpy())

predictions = np.asarray(predictions_list)
actual = y_val.numpy()

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
r2 = r2_score(actual, predictions)

print("\nRegularized Transformer Results")
print("--------------------------------")
print("Best epoch:", best_epoch)
print(f"Best validation MSE: {best_val_loss:.2f}")
print(f"MAE:  {mae:.2f} cycles")
print(f"RMSE: {rmse:.2f} cycles")
print(f"R²:   {r2:.4f}")

# Save separate files; preserve the original experiment
torch.save(
    model.state_dict(),
    MODEL_DIR / "rul_transformer_regularized.pt"
)

pd.DataFrame(history).to_csv(
    RESULTS_DIR / "transformer_regularized_history.csv",
    index=False
)

pd.DataFrame([{
    "model": "Regularized Transformer",
    "best_epoch": best_epoch,
    "best_validation_MSE": best_val_loss,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}]).to_csv(
    RESULTS_DIR / "transformer_regularized_metrics.csv",
    index=False
)

pd.DataFrame({
    "actual_RUL": actual,
    "predicted_RUL": predictions
}).to_csv(
    RESULTS_DIR / "transformer_regularized_predictions.csv",
    index=False
)

print("\nSaved regularized model and experiment logs.")
