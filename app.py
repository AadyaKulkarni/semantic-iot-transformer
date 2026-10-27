
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "semantic_model.pt"
DATA_PATH = ROOT / "data" / "sequences" / "validation_sequences.npz"
RESULTS_PATH = ROOT / "results" / "semantic_evaluation.csv"

st.set_page_config(
    page_title="Semantic IoT Communication",
    page_icon="📡",
    layout="wide"
)


class SemanticTransmitter(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(630, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        z = self.encoder(x.reshape(x.shape[0], -1))
        power = z.pow(2).mean(dim=1, keepdim=True)
        return z / torch.sqrt(power + 1e-8)


class SemanticReceiver(nn.Module):
    def __init__(self, latent_dim=8):
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


@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH, map_location="cpu", weights_only=True
    )

    transmitter = SemanticTransmitter(checkpoint["latent_dim"])
    receiver = SemanticReceiver(checkpoint["latent_dim"])

    transmitter.load_state_dict(checkpoint["transmitter"])
    receiver.load_state_dict(checkpoint["receiver"])

    transmitter.eval()
    receiver.eval()
    return transmitter, receiver, checkpoint["target_scale"]


@st.cache_data
def load_data():
    data = np.load(DATA_PATH)
    return data["X"].astype(np.float32), data["y"].astype(np.float32)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


st.title("📡 Transformer-Driven Semantic Communication for IoT")
st.caption(
    "Task-oriented sensor compression, simulated wireless transmission, "
    "and Remaining Useful Life (RUL) prediction."
)

if not MODEL_PATH.exists() or not DATA_PATH.exists() or not RESULTS_PATH.exists():
    st.error("Required model or data files are missing. Check your project folders.")
    st.stop()

transmitter, receiver, target_scale = load_model()
X, y = load_data()
results = load_results()

st.sidebar.header("Communication Settings")

snr = st.sidebar.select_slider(
    "Wireless channel SNR (dB)",
    options=[0, 5, 10, 20],
    value=10
)

sample_index = st.sidebar.number_input(
    "Validation sequence index",
    min_value=0,
    max_value=len(X) - 1,
    value=0,
    step=1
)

st.sidebar.caption(
    "Higher SNR means less simulated channel noise. "
    "The displayed prediction uses a newly sampled noise realization."
)

sequence = torch.from_numpy(X[int(sample_index):int(sample_index) + 1])

with torch.no_grad():
    latent = transmitter(sequence)

    if snr == 0:
        noise_std = 1.0
    else:
        noise_std = 10 ** (-snr / 20)

    # Fixed per selected sample/SNR for reproducible dashboard results.
    generator = torch.Generator().manual_seed(
        1000 + int(sample_index) * 10 + snr
    )
    noise = torch.randn(
        latent.shape, generator=generator, dtype=latent.dtype
    ) * noise_std

    received = latent + noise
    prediction = receiver(received).item() * target_scale

actual_rul = float(y[int(sample_index)])
latent_dim = latent.shape[1]
original_values = 30 * 21
reduction = (1 - latent_dim / original_values) * 100

st.subheader("Live Prediction")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted RUL", f"{max(0, prediction):.1f} cycles")
c2.metric("Actual RUL", f"{actual_rul:.1f} cycles")
c3.metric("Absolute Error", f"{abs(prediction - actual_rul):.1f} cycles")
c4.metric("SNR", f"{snr} dB")

st.subheader("Semantic Compression")

c1, c2, c3 = st.columns(3)
c1.metric("Original values", original_values)
c2.metric("Transmitted latent values", latent_dim)
c3.metric("Value-count reduction", f"{reduction:.2f}%")

st.info(
    "Value-count reduction is not the same as bit-rate reduction. "
    "This prototype uses floating-point latent values; quantization, "
    "bit encoding, and communication overhead are not yet included."
)


st.subheader("Validation Performance by SNR")

chart_data = results[results["SNR_dB"].notna()].copy()
chart_data["SNR_dB"] = chart_data["SNR_dB"].astype(int)
chart_data = chart_data.sort_values("SNR_dB")

# Use actual tested SNR values as categorical labels.
chart_data = chart_data.sort_values("SNR_dB")
chart_data["SNR"] = chart_data["SNR_dB"].astype(str) + " dB"
chart_data = chart_data.set_index("SNR")

left, right = st.columns(2)

with left:
    st.markdown("**Mean Absolute Error (MAE)**")
    st.line_chart(chart_data[["MAE"]], x_label="SNR", y_label="MAE (cycles)")

with right:
    st.markdown("**Root Mean Squared Error (RMSE)**")
    st.line_chart(chart_data[["RMSE"]], x_label="SNR", y_label="RMSE (cycles)")

st.caption("Lower MAE and RMSE indicate more accurate RUL predictions.")

st.subheader("Original Transformer vs. Semantic Model")

comparison = pd.DataFrame({
    "Model": ["Original Transformer", "Semantic model (no channel noise)"],
    "MAE (cycles)": [18.148, 17.593],
    "RMSE (cycles)": [25.2645, 24.287]
})

st.dataframe(comparison, use_container_width=True, hide_index=True)

st.caption(
    "These are validation results from separately trained models. "
    "The comparison is preliminary because the architectures and training "
    "procedures differ."
)


display_cols = ["condition", "MAE", "RMSE", "R2", "value_count_reduction_percent"]
st.dataframe(
    results[display_cols].round(3),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Metrics above are measured over the full validation set. "
    "The individual prediction uses the selected sequence and a simulated "
    "noise sample. This is a research prototype, not a live IoT deployment."
)
