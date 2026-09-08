import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load TEST data (IMPORTANT)
test = pd.read_csv("data/test.csv")
actual = test['Sales'].values

# Load predictions
def load_pred(name):
    try:
        return np.load(f"data/{name}_forecast.npy")
    except:
        return None

models = {
    "ARIMA": load_pred("arima"),
    "SARIMA": load_pred("sarima"),
    "Prophet": load_pred("prophet"),
    "LSTM": load_pred("lstm"),
    "XGBoost": load_pred("xgboost")
}

# Residual Analysis
for name, pred in models.items():
    if pred is None:
        continue

    min_len = min(len(actual), len(pred))
    actual_trim = actual[:min_len]
    pred_trim = pred[:min_len]

    residuals = actual_trim - pred_trim

    print(f"\n{name} Residual Summary")
    print("Mean:", np.mean(residuals))

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(residuals)
    plt.title(f"{name} Residual Plot")
    plt.axhline(0, linestyle='--')
    plt.show()