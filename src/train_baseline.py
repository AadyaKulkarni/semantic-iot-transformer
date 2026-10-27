
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Load the prepared CSV files
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

X_train = train_df[sensor_cols]
y_train = train_df["RUL"]

X_val = val_df[sensor_cols]
y_val = val_df["RUL"]

print("Training baseline model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predict RUL
predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
r2 = r2_score(y_val, predictions)

print("\nBaseline Results")
print("----------------")
print(f"MAE:  {mae:.2f} cycles")
print(f"RMSE: {rmse:.2f} cycles")
print(f"R²:   {r2:.4f}")

# Save model
model_path = MODELS_DIR / "baseline_random_forest.pkl"
joblib.dump(model, model_path)

# Save predictions for later comparisons
results = val_df[["engine_id", "cycle", "RUL"]].copy()
results["predicted_RUL"] = predictions
results.to_csv(RESULTS_DIR / "baseline_predictions.csv", index=False)

# Save metrics
pd.DataFrame([{
    "model": "Random Forest",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}]).to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)

print("\nSaved model to:", model_path)
print("Saved metrics and predictions to:", RESULTS_DIR)
