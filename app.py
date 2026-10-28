
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn


# --------------------------------------------------
# 1. PATHS AND PAGE CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "semantic_model_v1.pt"
DATA_PATH = ROOT / "data" / "sequences" / "validation_sequences.npz"
RESULTS_PATH = ROOT / "results" / "semantic_evaluation_v1.csv"

st.set_page_config(
    page_title="Semantic IoT Communication",
    page_icon="📡",
    layout="wide",
)

# Recorded validation results from previous experiments.
# These are not recalculated by this dashboard.
MODEL_RESULTS = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Original Transformer",
        "Regularized Transformer",
        "Semantic Communication (V1)",
    ],
    "MAE (cycles)": [25.77, 18.148, 18.48, 17.593],
    "RMSE (cycles)": [35.27, 25.2645, 25.46, 24.287],
    "R²": [0.7114, 0.8119, 0.8089, 0.8261],
})


# --------------------------------------------------
# 2. MODEL ARCHITECTURE
# Must match the architecture used during training.
# --------------------------------------------------

class SemanticTransmitter(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(630, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
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
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.decoder(z).squeeze(-1)


# --------------------------------------------------
# 3. LOAD MODEL, DATA AND RESULTS
# --------------------------------------------------

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    transmitter = SemanticTransmitter(checkpoint["latent_dim"])
    receiver = SemanticReceiver(checkpoint["latent_dim"])

    transmitter.load_state_dict(checkpoint["transmitter"])
    receiver.load_state_dict(checkpoint["receiver"])

    transmitter.eval()
    receiver.eval()

    return transmitter, receiver, float(checkpoint["target_scale"])


@st.cache_data
def load_data():
    with np.load(DATA_PATH) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

    return X, y


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


# --------------------------------------------------
# 4. CHECK REQUIRED FILES
# --------------------------------------------------

st.title("📡 Transformer-Driven Semantic Communication for IoT")

st.caption(
    "Task-oriented sensor compression, simulated wireless transmission, "
    "and Remaining Useful Life (RUL) prediction."
)

missing_files = [
    path.name
    for path in [MODEL_PATH, DATA_PATH, RESULTS_PATH]
    if not path.exists()
]

if missing_files:
    st.error(
        "Required files are missing: "
        + ", ".join(missing_files)
        + ". Check the models, data and results folders."
    )
    st.stop()

try:
    transmitter, receiver, target_scale = load_model()
    X, y = load_data()
    results = load_results()
except Exception as exc:
    st.error(f"Could not load the project files: {exc}")
    st.stop()

if X.ndim != 3 or X.shape[1:] != (30, 21):
    st.error(
        f"Expected input sequences with shape (samples, 30, 21), "
        f"but received {X.shape}."
    )
    st.stop()

if len(X) != len(y) or len(X) == 0:
    st.error("The validation data is empty or X and y have different lengths.")
    st.stop()


# --------------------------------------------------
# 5. SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("Communication Settings")

snr = st.sidebar.select_slider(
    "Wireless channel SNR (dB)",
    options=[0, 5, 10, 20],
    value=10,
)

sample_index = st.sidebar.number_input(
    "Validation sequence index",
    min_value=0,
    max_value=len(X) - 1,
    value=0,
    step=1,
)

st.sidebar.caption(
    "Higher SNR means less simulated channel noise. "
    "The selected sample and SNR produce reproducible noise."
)


# --------------------------------------------------
# 6. SEMANTIC ENCODING, CHANNEL AND RUL PREDICTION
# --------------------------------------------------

sequence = torch.from_numpy(
    X[int(sample_index):int(sample_index) + 1]
)

with torch.no_grad():
    latent = transmitter(sequence)

    noise_std = 10 ** (-snr / 20)

    # Reproducible simulated AWGN for the selected sample and SNR.
    generator = torch.Generator().manual_seed(
        1000 + int(sample_index) * 10 + int(snr)
    )

    noise = torch.randn(
        latent.shape,
        generator=generator,
        dtype=latent.dtype,
    ) * noise_std

    received = latent + noise
    prediction = receiver(received).item() * target_scale

actual_rul = float(y[int(sample_index)])
absolute_error = abs(prediction - actual_rul)

latent_dim = int(latent.shape[1])
original_values = int(np.prod(X.shape[1:]))
reduction = (1 - latent_dim / original_values) * 100


# --------------------------------------------------
# 7. LIVE PREDICTION
# --------------------------------------------------

st.header("1. Live RUL Prediction")

st.caption(
    f"Validation sample {int(sample_index)} | "
    "Prediction generated by the V1 model at the selected SNR."
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Predicted RUL", f"{max(0, prediction):.1f} cycles")
c2.metric("Actual RUL", f"{actual_rul:.1f} cycles")
c3.metric("Absolute Error", f"{absolute_error:.1f} cycles")
c4.metric("Channel SNR", f"{snr} dB")

st.caption(
    "The predicted RUL is clipped at zero for display only. "
    "Absolute error is calculated using the raw model prediction."
)


# --------------------------------------------------
# 8. SEMANTIC COMPRESSION
# --------------------------------------------------

st.header("2. Semantic Compression")

st.write(
    "The encoder maps a window of multivariate sensor readings "
    "into a compact latent representation for the simulated channel."
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Original sensor values",
    f"{original_values}",
    help="30 time steps × 21 features.",
)

c2.metric(
    "Latent values",
    f"{latent_dim}",
    help="Number of values output by the semantic encoder.",
)

c3.metric(
    "Value-count reduction",
    f"{reduction:.2f}%",
    help="Reduction in number of values, not measured bit-rate or energy savings.",
)

st.progress(max(0.0, min(1.0, reduction / 100)))

# Visual comparison of the representation sizes.
compression_chart = pd.DataFrame({
    "Representation": ["Original input", "Semantic representation"],
    "Number of values": [original_values, latent_dim],
}).set_index("Representation")

st.bar_chart(compression_chart)

st.success(
    f"The representation changes from {original_values} values to "
    f"{latent_dim} values: {reduction:.2f}% fewer values."
)

st.warning(
    "This is value-count reduction, not a measured bit-rate reduction. "
    "The prototype does not yet implement latent quantization, bit encoding, "
    "protocol overhead measurement, physical wireless transmission, "
    "or energy measurement."
)


# --------------------------------------------------
# 9. END-TO-END COMMUNICATION PIPELINE
# --------------------------------------------------

st.header("3. End-to-End Communication Pipeline")

p1, p2, p3, p4 = st.columns(4)

p1.metric("Sensor input", "30 × 21", "630 values")
p2.metric("Semantic encoder", f"{original_values} → {latent_dim}", "Compact representation")
p3.metric("Simulated channel", f"{snr} dB", "AWGN noise")
p4.metric("Receiver output", f"{max(0, prediction):.1f}", "Predicted RUL cycles")

with st.expander("View technical details for the selected sample"):
    latent_array = latent.squeeze(0).cpu().numpy()
    received_array = received.squeeze(0).cpu().numpy()

    latent_df = pd.DataFrame({
        "Latent index": np.arange(1, latent_dim + 1),
        "Transmitted value": latent_array,
        "Received value": received_array,
    })

    st.dataframe(
        latent_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The table shows the encoder output and the same representation "
        "after simulated channel noise. Values are floating-point numbers."
    )


# --------------------------------------------------
# 10. MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

st.header("4. Model Performance Comparison")

st.write(
    "Compare the previously recorded validation metrics of the baseline "
    "models and the proposed semantic communication model."
)

st.dataframe(
    MODEL_RESULTS.round(4),
    use_container_width=True,
    hide_index=True,
)

left, right = st.columns(2)

with left:
    st.markdown("**MAE by Model (lower is better)**")
    st.bar_chart(
        MODEL_RESULTS.set_index("Model")[["MAE (cycles)"]]
    )

with right:
    st.markdown("**RMSE by Model (lower is better)**")
    st.bar_chart(
        MODEL_RESULTS.set_index("Model")[["RMSE (cycles)"]]
    )

st.markdown("**R² by Model (higher is better)**")
st.bar_chart(
    MODEL_RESULTS.set_index("Model")[["R²"]]
)

st.caption(
    "These are previously recorded validation results, not freshly "
    "recalculated by this dashboard. The comparison is preliminary because "
    "the architectures and training procedures differ. Confirm consistent "
    "data splits and evaluation conditions before making strong claims."
)


# --------------------------------------------------
# 11. SEMANTIC MODEL PERFORMANCE BY SNR
# --------------------------------------------------

st.header("5. Channel Robustness Analysis")

st.write(
    "The following metrics show how the semantic model's validation "
    "performance changes as the simulated channel noise changes."
)

required_columns = {"MAE", "RMSE", "SNR_dB"}

if required_columns.issubset(results.columns):
    chart_data = results.copy()
    chart_data["SNR_dB"] = pd.to_numeric(
        chart_data["SNR_dB"], errors="coerce"
    )
    chart_data["MAE"] = pd.to_numeric(
        chart_data["MAE"], errors="coerce"
    )
    chart_data["RMSE"] = pd.to_numeric(
        chart_data["RMSE"], errors="coerce"
    )

    chart_data = chart_data.dropna(
        subset=["SNR_dB", "MAE", "RMSE"]
    ).sort_values("SNR_dB")

    # Only show rows for channel SNR experiments.
    chart_data = chart_data[chart_data["SNR_dB"] >= 0]
    chart_data["SNR"] = (
        chart_data["SNR_dB"].astype(int).astype(str) + " dB"
    )

    if not chart_data.empty:
        chart_data = chart_data.set_index("SNR")

        left, right = st.columns(2)

        with left:
            st.markdown("**MAE vs. SNR (lower is better)**")
            st.line_chart(
                chart_data[["MAE"]],
                y_label="MAE (cycles)",
            )

        with right:
            st.markdown("**RMSE vs. SNR (lower is better)**")
            st.line_chart(
                chart_data[["RMSE"]],
                y_label="RMSE (cycles)",
            )
    else:
        st.warning(
            "No valid SNR evaluation rows were found in the results CSV."
        )
else:
    st.warning(
        "The evaluation CSV does not contain the expected SNR_dB, MAE "
        "and RMSE columns. Check its column names."
    )


# --------------------------------------------------
# 12. EVALUATION RESULTS TABLE
# --------------------------------------------------

st.header("6. Detailed Validation Results")

display_cols = [
    col for col in [
        "condition",
        "SNR_dB",
        "MAE",
        "RMSE",
        "R2",
        "value_count_reduction_percent",
    ]
    if col in results.columns
]

if display_cols:
    st.dataframe(
        results[display_cols].round(3),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No recognized evaluation columns were found in the CSV.")


# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Research prototype using NASA C-MAPSS validation data. "
    "The wireless channel is simulated with additive white Gaussian noise "
    "(AWGN). The dashboard does not represent a physical IoT deployment."
)
