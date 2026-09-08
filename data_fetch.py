import requests
import pandas as pd
import numpy as np

url = "https://fakestoreapi.com/products"
data = requests.get(url).json()

df_products = pd.DataFrame(data)

dates = pd.date_range(start="2020-01-01", end="2025-12-31")

sales_data = []

for _, row in df_products.iterrows():
    for date in dates:
        sales = np.random.randint(10, 1000)
        
        # Seasonality
        if date.month in [11, 12]:
            sales *= 1.5
        
        revenue = sales * row['price']
        cost = revenue * 0.7
        profit = revenue - cost

        sales_data.append([
            date, row['title'], row['category'],
            sales, revenue, cost, profit
        ])

df = pd.DataFrame(sales_data, columns=[
    "Date", "Product", "Category",
    "Sales", "Revenue", "Cost", "Profit"
])

df.to_csv("data/sales_data.csv", index=False)
print("Data Saved ✅")