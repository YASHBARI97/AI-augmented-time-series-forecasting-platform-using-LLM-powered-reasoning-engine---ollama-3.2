import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings for a clean industry-level terminal output
warnings.filterwarnings("ignore")

# 1. Load Data (index_col=0 ensures Date is the index)
train = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
test = pd.read_csv("data/test.csv", index_col=0, parse_dates=True)

# Ensure frequency is set to Daily
train_series = train['Sales'].asfreq('D')

# 2. Grid Search Candidates
candidates = [(1,0,0), (0,0,1), (1,0,1), (2,0,1), (2,0,2), (3,0,1), (1,0,2), (1,1,1)]
model_results = []

print("🚀 Starting ARIMA Grid Search (AIC/BIC Optimization)...")

for order in candidates:
    try:
        model = ARIMA(train_series, order=order)
        res = model.fit()
        model_results.append({
            'order': order,
            'aic': res.aic,
            'bic': res.bic,
            'model_obj': res
        })
        print(f"Checked ARIMA{order}: AIC={res.aic:.2f}, BIC={res.bic:.2f}")
    except:
        continue

# 3. Select Best Model (Comparing AIC and BIC - BIC is safer for stability)
stats_df = pd.DataFrame(model_results)
best_model_row = stats_df.loc[stats_df['bic'].idxmin()]
best_res = best_model_row['model_obj']

print(f"\n✅ Best Order Selected: ARIMA{best_model_row['order']}")
print(f"📊 Final AIC: {best_model_row['aic']:.2f} | Final BIC: {best_model_row['bic']:.2f}")

# 4. Forecast
forecast = best_res.forecast(steps=len(test))

# Save
np.save("data/arima_forecast.npy", forecast)
print("💾 Forecast saved to data/arima_forecast.npy")