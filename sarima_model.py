import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
test = pd.read_csv("data/test.csv", index_col=0, parse_dates=True)
train_series = train['Sales'].asfreq('D')

# 2. Optimized Candidates (p,d,q) x (P,D,Q,s)
# s=7 for Weekly patterns, s=12 for Monthly. Using 7 for sales behavior.
candidates_sarima = [
    ((1,1,1), (1,1,1,7)),
    ((2,1,0), (1,1,0,7)),
    ((1,0,1), (0,1,1,7))
]

results_sarima = []

print("🚀 Starting SARIMA Grid Search...")

for order, seasonal in candidates_sarima:
    try:
        model = SARIMAX(train_series, order=order, seasonal_order=seasonal)
        # Using L-BFGS for better convergence in SARIMA
        res = model.fit(disp=False, maxiter=200, method='lbfgs')
        results_sarima.append({
            'config': (order, seasonal),
            'aic': res.aic,
            'res_obj': res
        })
        print(f"Checked SARIMA{order}{seasonal}: AIC={res.aic:.2f}")
    except:
        continue

# 3. Select Best based on AIC
best_sarima_row = min(results_sarima, key=lambda x: x['aic'])
best_res_sarima = best_sarima_row['res_obj']

print(f"\n✅ Best SARIMA Config: {best_sarima_row['config']}")

# 4. Forecast
forecast = best_res_sarima.forecast(steps=len(test))

# Save
np.save("data/sarima_forecast.npy", forecast)
print("💾 Forecast saved to data/sarima_forecast.npy")