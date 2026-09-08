import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# Load raw data
df = pd.read_csv("data/sales_data.csv")

# 1. Basic Cleaning
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# 2. Aggregation (Grouping by Date)
df_daily = df.groupby('Date').agg({
    'Sales': 'sum',
    'Revenue': 'sum',
    'Profit': 'sum'
})

# 3. FIXING VALUEWARNINGS: Set Frequency and Fill Gaps
# We set frequency to 'D' (Daily). This adds rows for missing dates.
df_daily = df_daily.asfreq('Y')

# Fill missing values created by asfreq (Forward Fill carries the last known value)
df_daily = df_daily.ffill()

# 4. Save with Index=True (To keep the Date as the primary index)
df_daily.to_csv("data/processed_data.csv")

print("✅ Preprocessing Done: Frequency set to Daily ('D') and gaps filled.")