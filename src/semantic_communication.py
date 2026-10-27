
import numpy as np
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sequences" / "validation_sequences.npz"
RESULTS_PATH = ROOT / "results"
RESULTS_PATH.mkdir(exist_ok=True)

# Communication settings
LATENT_DIM = 8
SNR_LEVELS = [0, 5, 10, 20]


def compress_sequence(sequence):
    """
    Simple baseline compressor using temporal averaging and
    linear projection. This is a prototype, not a learned
    semantic encoder.
    """
    # sequence shape: (30, 21)
    temporal_summary = sequence.mean(axis=0)

    rng = np.random.default_rng(42)
    projection = rng.normal(
        0, 1 / np.sqrt(sequence.shape[1]),
        size=(sequence.shape[1], LATENT_DIM)
    )

    latent = temporal_summary @ projection
    return latent.astype(np.float32)


def awgn_channel(signal, snr_db, rng):
    """Transmit a real-valued vector through an AWGN channel."""
    signal = np.asarray(signal, dtype=np.float32)
    signal_power = np.mean(signal ** 2)

    if signal_power < 1e-12:
        return signal.copy()

    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(
        0, np.sqrt(noise_power), size=signal.shape
    )
    return (signal + noise).astype(np.float32)


def main():
    data = np.load(DATA_PATH)
    X = data["X"]
    y = data["y"]

    print("Validation sequences:", X.shape)
    print("Validation labels:", y.shape)

    # Use one sequence for the first communication test
    sequence = X[0]
    latent = compress_sequence(sequence)

    original_values = sequence.size
    compressed_values = latent.size
    reduction = 1 - compressed_values / original_values

    print("\n--- Compression ---")
    print("Original values:", original_values)
    print("Compressed values:", compressed_values)
    print(f"Value-count reduction: {reduction * 100:.2f}%")
    print("\n--- AWGN Channel Test ---")

    rng = np.random.default_rng(123)

    rows = []
    for snr in SNR_LEVELS:
        received = awgn_channel(latent, snr, rng)
        noise_mse = float(np.mean((received - latent) ** 2))

        rows.append({
            "SNR_dB": snr,
            "Latent_Dimension": len(latent),
            "Noise_MSE": noise_mse,
        })

        print(
            f"SNR {snr:>2} dB | "
            f"Received dimension: {len(received)} | "
            f"Noise MSE: {noise_mse:.6f}"
        )

    import pandas as pd

    output = RESULTS_PATH / "channel_test_results.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()
