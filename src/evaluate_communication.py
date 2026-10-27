
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sequences" / "validation_sequences.npz"
MODEL_PATH = ROOT / "models" / "rul_transformer.pt"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LATENT_DIM = 8
SNR_LEVELS = [0, 5, 10, 20]
DEVICE = "cpu"


class RULTransformer(nn.Module):
    def __init__(self, input_dim=21, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, 30, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
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
        return self.regressor(x[:, -1, :]).squeeze(-1)


def compress_sequence(sequence, projection):
    # Average over time, then project 21 sensor values to 8 values.
    return (sequence.mean(axis=0) @ projection).astype(np.float32)


def reconstruct_sequence(latent, projection):
    # Approximate inverse projection and repeat across 30 time steps.
    reconstructed_sensors = latent @ np.linalg.pinv(projection)
    return np.tile(reconstructed_sensors, (30, 1)).astype(np.float32)


def awgn_channel(signal, snr_db, rng):
    power = np.mean(signal ** 2)
    if power < 1e-12:
        return signal.copy()

    noise_power = power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape)
    return (signal + noise).astype(np.float32)


def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }


def main():
    data = np.load(DATA_PATH)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    model = RULTransformer().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    rng = np.random.default_rng(42)
    projection = rng.normal(
        0, 1 / np.sqrt(21), size=(21, LATENT_DIM)
    ).astype(np.float32)

    # Original, uncompressed reference
    with torch.no_grad():
        original_predictions = model(torch.from_numpy(X)).numpy()

    original_metrics = calculate_metrics(y, original_predictions)
    results = [{
        "condition": "Original",
        "SNR_dB": np.nan,
        "MAE": original_metrics["MAE"],
        "RMSE": original_metrics["RMSE"],
        "R2": original_metrics["R2"],
        "latent_values": 630,
        "value_reduction_percent": 0.0
    }]

    print("Original input metrics:")
    print({k: round(v, 4) for k, v in original_metrics.items()})

    # Compressed and noisy transmission
    for snr in SNR_LEVELS:
        predictions = []

        for sequence in X:
            latent = compress_sequence(sequence, projection)
            received = awgn_channel(latent, snr, rng)
            reconstructed = reconstruct_sequence(received, projection)

            with torch.no_grad():
                pred = model(torch.from_numpy(reconstructed[None, :, :]))
                predictions.append(pred.item())

        metrics = calculate_metrics(y, np.array(predictions))
        results.append({
            "condition": "Compressed + AWGN",
            "SNR_dB": snr,
            **metrics,
            "latent_values": LATENT_DIM,
            "value_reduction_percent": 100 * (1 - LATENT_DIM / 630)
        })

        print(
            f"SNR {snr:>2} dB | MAE: {metrics['MAE']:.2f} | "
            f"RMSE: {metrics['RMSE']:.2f} | R2: {metrics['R2']:.4f}"
        )

    output = RESULTS_DIR / "communication_evaluation.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    print("\nSaved results to:", output)


if __name__ == "__main__":
    main()
