import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

warnings.filterwarnings('ignore')

def run_stationarity_tests():
    # Load data
    df = pd.read_csv("data/processed_data.csv")
    series = df['Sales'].dropna()

    # ADF Test
    adf_result = adfuller(series)

    # KPSS Test
    # Added regression='c' explicitly
    kpss_result = kpss(series, regression='c')

    results = {
        "ADF Statistic": round(adf_result[0], 4),
        "ADF p-value": round(adf_result[1], 4),
        "KPSS Statistic": round(kpss_result[0], 4),
        "KPSS p-value": round(kpss_result[1], 4)
    }

    return results