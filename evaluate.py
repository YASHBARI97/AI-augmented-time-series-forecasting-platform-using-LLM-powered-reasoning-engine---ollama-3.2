import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load actual data
df = pd.read_csv("data/processed_data.csv")

# Take last 30 actual values for comparison
actual = df['Sales'].tail(30).values

def evaluate_model(name):
    try:
        pred = np.load(f"data/{name}_forecast.npy")

        # Flatten LSTM predictions if needed
        if len(pred.shape) > 1:
            pred = pred.flatten()

        # Align lengths
        min_len = min(len(actual), len(pred))
        actual_trim = actual[:min_len]
        pred_trim = pred[:min_len]

        # Metrics
        mse = mean_squared_error(actual_trim, pred_trim)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_trim, pred_trim)

        print(f"\n📊 {name.upper()} MODEL PERFORMANCE")
        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"MAE  : {mae:.2f}")

    except Exception as e:
        print(f"\n❌ {name.upper()} not evaluated:", e)

# List of models
models = ["arima", "sarima", "prophet", "lstm"]

for model in models:
    evaluate_model(model)