import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

# Load TEST data
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

# Ljung-Box Test
for name, pred in models.items():
    if pred is None:
        continue

    min_len = min(len(actual), len(pred))
    actual_trim = actual[:min_len]
    pred_trim = pred[:min_len]

    residuals = actual_trim - pred_trim

    result = acorr_ljungbox(residuals, lags=[10], return_df=True)

    print(f"\n{name} Ljung-Box Test")
    print(result)