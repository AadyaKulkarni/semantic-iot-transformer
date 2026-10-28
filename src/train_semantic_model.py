
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

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "data" / "sequences"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"
LATENT_DIM = 8
EPOCHS = 40
PATIENCE = 7
BATCH_SIZE = 128
LEARNING_RATE = 0.0003
TARGET_SCALE = 125.0
SNR_LEVELS = [0, 5, 10, 20]


class SemanticTransmitter(nn.Module):
    """Compress the sensor sequence into an 8-value task-oriented message."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30 * 21, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        z = self.encoder(x.reshape(x.shape[0], -1))

        # Normalize message power to approximately one per sample.
        power = z.pow(2).mean(dim=1, keepdim=True)
        z = z / torch.sqrt(power + 1e-8)
        return z


class SemanticReceiver(nn.Module):
    """Predict RUL directly from the received compact message."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        return self.decoder(z).squeeze(-1)


def add_awgn(z, snr_db):
    """Add Gaussian noise to a unit-power latent message."""
    noise_std = 10 ** (-snr_db / 20)
    return z + torch.randn_like(z) * noise_std


def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


def main():
    train_data = np.load(SEQ_DIR / "train_sequences.npz")
    val_data = np.load(SEQ_DIR / "validation_sequences.npz")

    X_train = torch.tensor(train_data["X"], dtype=torch.float32)
    y_train = torch.tensor(train_data["y"], dtype=torch.float32) / TARGET_SCALE
    X_val = torch.tensor(val_data["X"], dtype=torch.float32)
    y_val = val_data["y"].astype(np.float32)

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    transmitter = SemanticTransmitter().to(DEVICE)
    receiver = SemanticReceiver().to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(transmitter.parameters()) + list(receiver.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

    best_loss = float("inf")
    best_transmitter = None
    best_receiver = None
    patience_left = PATIENCE
    history = []

    print("Training task-oriented semantic communication model")
    print("Training sequences:", len(X_train))
    print("Latent dimension:", LATENT_DIM)

    for epoch in range(1, EPOCHS + 1):
        transmitter.train()
        receiver.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            # Train across different channel conditions.
            snr = random.choice(SNR_LEVELS)
            z = transmitter(xb)
            received = add_awgn(z, snr)
            prediction = receiver(received)

            loss = criterion(prediction, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(X_train)

        # Evaluate without channel noise for checkpoint selection.
        transmitter.eval()
        receiver.eval()
        with torch.no_grad():
            z_val = transmitter(X_val)
            val_prediction = receiver(z_val).cpu().numpy() * TARGET_SCALE

        val_mse = mean_squared_error(y_val, val_prediction)
        scheduler.step(val_mse)
        history.append({
            "epoch": epoch,
            "train_mse_scaled": train_loss,
            "validation_mse_cycles": val_mse
        })

        print(
            f"Epoch {epoch:02}/{EPOCHS} | "
            f"Train scaled MSE: {train_loss:.4f} | "
            f"Validation MSE: {val_mse:.2f}"
        )

        if val_mse < best_loss:
            best_loss = val_mse
            best_transmitter = copy.deepcopy(transmitter.state_dict())
            best_receiver = copy.deepcopy(receiver.state_dict())
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break

    transmitter.load_state_dict(best_transmitter)
    receiver.load_state_dict(best_receiver)

    torch.save({
        "transmitter": transmitter.state_dict(),
        "receiver": receiver.state_dict(),
        "latent_dim": LATENT_DIM,
        "target_scale": TARGET_SCALE
    }, MODEL_DIR / "semantic_model_v2.pt")

    pd.DataFrame(history).to_csv(
       RESULTS_DIR / "semantic_training_history_v2.csv", index=False
    )

    # Final evaluation with consistent validation sequences.
    rows = []
    transmitter.eval()
    receiver.eval()

    with torch.no_grad():
        z_val = transmitter(X_val)

        conditions = [("No channel noise", None)] + [
            (f"{snr} dB", snr) for snr in SNR_LEVELS
        ]

        for name, snr in conditions:
            if snr is None:
                received = z_val
            else:
                torch.manual_seed(SEED + snr)
                received = add_awgn(z_val, snr)

            predictions = (
                receiver(received).cpu().numpy() * TARGET_SCALE
            )
            result = metrics(y_val, predictions)

            rows.append({
                "condition": name,
                "SNR_dB": snr if snr is not None else np.nan,
                "latent_values": LATENT_DIM,
                "original_values": 30 * 21,
                "value_count_reduction_percent":
                    100 * (1 - LATENT_DIM / (30 * 21)),
                **result
            })

            print(
                f"{name:>16} | MAE: {result['MAE']:.2f} | "
                f"RMSE: {result['RMSE']:.2f} | "
                f"R2: {result['R2']:.4f}"
            )

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "semantic_evaluation_v2.csv", index=False
    )

    print("\nSaved model:", MODEL_DIR / "semantic_model_v2.pt")
    print("Saved training history:", RESULTS_DIR / "semantic_training_history_v2.csv")
    print("Saved evaluation:", RESULTS_DIR / "semantic_evaluation_v2.csv")


if __name__ == "__main__":
    main()
