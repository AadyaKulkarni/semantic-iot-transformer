# Transformer-Driven Semantic Communication for IoT

A task-oriented semantic communication prototype that uses a lightweight neural encoder and receiver to predict Remaining Useful Life (RUL) from turbofan engine sensor data under simulated wireless channel noise.

## Overview

Traditional communication systems aim to transmit data accurately. Semantic communication instead focuses on preserving information relevant to a specific task.

This project explores task-oriented communication for predictive maintenance. Sensor time-series data from the NASA C-MAPSS turbofan engine dataset is used to develop an RUL prediction model and a learned semantic communication pipeline.

## Key Features

- RUL prediction using a Transformer model.
- Random Forest baseline for comparison.
- Learned semantic transmitter that compresses sensor sequences into an 8-value latent representation.
- Simulated Additive White Gaussian Noise (AWGN) channel.
- Evaluation under different Signal-to-Noise Ratio (SNR) levels.
- Streamlit dashboard for interactive visualization.
- Evaluation using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

## System Architecture

```text
Sensor Time-Series Data
          |
          v
   Preprocessing
          |
          v
  Semantic Transmitter
   (630 values -> 8)
          |
          v
    AWGN Channel
  (0, 5, 10, 20 dB)
          |
          v
   Semantic Receiver
          |
          v
      RUL Prediction
          |
          v
   Performance Metrics
```

## Dataset

This project uses the NASA C-MAPSS Turbofan Engine Degradation Simulation dataset.

Dataset information: [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

The dataset is not included in this repository. Download it separately and place the required FD001 files in the expected local dataset directory before running preprocessing scripts.

## Project Structure

```text
semantic-iot-transformer/
├── app.py
├── src/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── preprocess_data.py
│   ├── prepare_sequences.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── train_transformer_regularized.py
│   ├── semantic_communication.py
│   ├── evaluate_communication.py
│   ├── train_semantic_model.py
│   └── check_model.py
├── models/
│   ├── rul_transformer.pt
│   └── semantic_model.pt
├── results/
│   ├── semantic_evaluation.csv
│   ├── semantic_training_history.csv
│   ├── transformer_metrics.csv
│   ├── baseline_metrics.csv
│   ├── channel_test_results.csv
│   └── plots/
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 or newer is recommended.

1. Clone the repository:

   ```bash
   git clone https://github.com/AadyaKulkarni/semantic-iot-transformer.git
   cd semantic-iot-transformer
   ```

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Prepare the dataset and required sequence files using the scripts in `src/`.

5. Launch the dashboard:

   ```bash
   python -m streamlit run app.py
   ```

## Current Results

The current validation experiments produced the following results:

| Model / Condition | MAE (cycles) | RMSE (cycles) | R² |
|---|---:|---:|---:|
| Original Transformer | 18.15 | 25.26 | 0.8119 |
| Semantic model, no channel noise | 17.59 | 24.29 | 0.8261 |
| Semantic model, 0 dB SNR | 23.35 | 30.70 | 0.7222 |
| Semantic model, 5 dB SNR | 19.25 | 25.80 | 0.8038 |
| Semantic model, 10 dB SNR | 18.30 | 24.95 | 0.8165 |
| Semantic model, 20 dB SNR | 17.65 | 24.34 | 0.8253 |

These are preliminary results from separate models and a simulated channel. The comparison is not a controlled claim that semantic communication outperforms the original Transformer.

## Limitations

- Wireless transmission is simulated; no physical IoT hardware is used.
- The current latent representation uses floating-point values. Quantization, bit-level encoding, protocol overhead, and actual bandwidth savings have not yet been measured.
- Evaluation uses the FD001 dataset and a validation split; broader testing is needed before generalizing results.
- Model performance may vary with channel conditions and dataset characteristics.

## Technologies

- Python
- PyTorch
- NumPy and Pandas
- scikit-learn
- Matplotlib
- Streamlit

## Project Status

Academic prototype in development for a Student Development Project (SDP).

## License

No license has been added yet. Contact the repository owner before reusing or redistributing this project's code.
