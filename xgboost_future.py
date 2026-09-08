import pandas as pd
import numpy as np
from xgboost import XGBRegressor

def generate_xgboost_forecast(days_to_forecast=30):
    # 1. Load Data
    df = pd.read_csv("data/processed_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    # 2. Feature Engineering (Lags + Date Parts)
    df['lag1'] = df['Sales'].shift(1)
    df['lag2'] = df['Sales'].shift(2)
    df['lag3'] = df['Sales'].shift(3)
    
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['year'] = df['Date'].dt.year
    
    df = df.dropna()

    # 3. Train on Full Data
    features = ['lag1', 'lag2', 'lag3', 'month', 'day', 'year']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(df[features], df['Sales'])

    # 4. Recursive Forecasting Engine
    forecast = []
    future_dates = []
    
    # Correct Seeding: Use the absolute last 3 points of sales
    l1, l2, l3 = df['Sales'].iloc[-1], df['Sales'].iloc[-2], df['Sales'].iloc[-3]
    last_date = df['Date'].iloc[-1]

    for i in range(days_to_forecast):
        next_date = last_date + pd.Timedelta(days=1)
        
        # Prepare Input
        input_data = pd.DataFrame([[l1, l2, l3, next_date.month, next_date.day, next_date.year]], 
                                 columns=features)
        
        # Predict
        pred = model.predict(input_data)[0]
        
        forecast.append(pred)
        future_dates.append(next_date)

        # Update recursive lags for next iteration
        l3, l2, l1 = l2, l1, pred
        last_date = next_date

    return pd.DataFrame({"Date": future_dates, "Predicted_Sales": forecast})

# To use it:
# result = generate_xgboost_forecast(30)
# result.to_csv("data/xgboost_future.csv", index=False)