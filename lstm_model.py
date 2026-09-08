import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1. LOAD & SCALE DATA (Critical for LSTM)
# -----------------------------
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[['Sales']])
test_scaled = scaler.transform(test_df[['Sales']])

# -----------------------------
# 2. INCREASE WINDOW SIZE
# -----------------------------
# Increasing from 10 to 30 allows the model to see monthly patterns
def create_sequences(data, window=30):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, window=30)
X_test, y_test = create_sequences(test_scaled, window=30)

# -----------------------------
# 3. DEEPER ARCHITECTURE
# -----------------------------
model = Sequential()
# Layer 1: Return sequences to pass data to the next LSTM layer
model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(30, 1)))
model.add(Dropout(0.2)) 

# Layer 2: Standard LSTM layer
model.add(LSTM(32, activation='tanh'))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# -----------------------------
# 4. TRAIN & INVERSE TRANSFORM
# -----------------------------
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# Predict
pred_scaled = model.predict(X_test)

# Convert back to original scale (Sales units)
pred = scaler.inverse_transform(pred_scaled).flatten()

# Save
np.save("data/lstm_forecast_v2.npy", pred)
print("✅ Optimized LSTM Forecast Saved")