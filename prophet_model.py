from prophet import Prophet
import pandas as pd
import numpy as np

# Load split data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Convert to Prophet format
train = train.rename(columns={'Date': 'ds', 'Sales': 'y'})
test = test.rename(columns={'Date': 'ds', 'Sales': 'y'})

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = Prophet()
model.fit(train)

# -----------------------------
# FORECAST ON TEST DATA
# -----------------------------
future = test[['ds']]   # IMPORTANT: only test dates

forecast = model.predict(future)

# Save predictions
np.save("data/prophet_forecast.npy", forecast['yhat'].values)

print("✅ Prophet Test Forecast Saved")