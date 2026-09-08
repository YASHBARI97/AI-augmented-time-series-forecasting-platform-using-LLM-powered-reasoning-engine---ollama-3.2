import pandas as pd

# Load processed data
# index_col=0 ensures the 'Date' column stays as the index
df = pd.read_csv("data/processed_data.csv", index_col=0, parse_dates=True)

# Time-based split (80% Train, 20% Test)
split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

# Save
# index=True is vital for ARIMA/SARIMA to see the frequency info
train.to_csv("data/train.csv", index=True)
test.to_csv("data/test.csv", index=True)

print(f"✅ Train/Test split saved (Train: {len(train)}, Test: {len(test)})")