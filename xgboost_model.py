import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# -----------------------------
# LOAD SPLIT DATA
# -----------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

# -----------------------------
# FEATURE ENGINEERING FUNCTION
# -----------------------------
def create_features(df):
    df['lag1'] = df['Sales'].shift(1)
    df['lag2'] = df['Sales'].shift(2)
    df['lag3'] = df['Sales'].shift(3)

    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['year'] = df['Date'].dt.year

    return df

# Apply features
train = create_features(train)
test = create_features(test)

# Drop nulls
train = train.dropna()
test = test.dropna()

# -----------------------------
# FEATURES & TARGET
# -----------------------------
features = ['lag1', 'lag2', 'lag3', 'month', 'day', 'year']

X_train = train[features]
y_train = train['Sales']

X_test = test[features]
y_test = test['Sales']

# -----------------------------
# MODEL
# -----------------------------
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)

model.fit(X_train, y_train)

# -----------------------------
# PREDICTION (ON TEST DATA)
# -----------------------------
forecast = model.predict(X_test)

# Save forecast
np.save("data/xgboost_forecast.npy", forecast)

print("✅ XGBoost Test Prediction Saved")