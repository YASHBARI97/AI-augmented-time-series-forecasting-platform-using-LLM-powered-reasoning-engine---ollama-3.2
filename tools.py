"""
ai/tools.py
-----------
Clean, callable wrappers around the existing forecasting pipeline
(data loading, model comparison, diagnostics, forecasting).

These functions serve two purposes:
  1. Reused directly by Streamlit sections (Overview, Evaluation, etc.)
  2. Exposed to Claude as "tools" so the AI Chat / Agent layer can call
     them autonomously based on natural-language questions.

Each function returns plain Python data (dict / list / DataFrame-as-dict)
so it is easy to both render in Streamlit and pass back into Claude as
tool_result content.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR = "data"
MODEL_NAMES = ["arima", "sarima", "prophet", "lstm", "xgboost"]


# -----------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------
def load_base_data() -> pd.DataFrame:
    df = pd.read_csv(f"{DATA_DIR}/processed_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_forecast(name: str):
    try:
        data = np.load(f"{DATA_DIR}/{name}_forecast.npy")
        return data.flatten()
    except Exception:
        return None


def load_all_forecasts() -> dict:
    return {name: load_forecast(name) for name in MODEL_NAMES}


# -----------------------------------------------------------------------
# TOOL: get_model_metrics
# -----------------------------------------------------------------------
def get_model_metrics(test_window: int = 30) -> list[dict]:
    """
    Returns RMSE / MSE / MAE for every available model, computed against
    the most recent `test_window` actual data points.
    """
    df = load_base_data()
    actual_test = df["Sales"].tail(test_window).values
    forecasts = load_all_forecasts()

    results = []
    for name, pred in forecasts.items():
        if pred is None:
            results.append({"model": name.upper(), "mse": None, "rmse": None, "mae": None})
            continue
        n = min(len(pred), len(actual_test))
        p, a = pred[:n], actual_test[:n]
        mse = float(mean_squared_error(a, p))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(a, p))
        results.append({"model": name.upper(), "mse": round(mse, 2), "rmse": round(rmse, 2), "mae": round(mae, 2)})
    return results


# -----------------------------------------------------------------------
# TOOL: get_stationarity_report
# -----------------------------------------------------------------------
def get_stationarity_report() -> dict:
    df = load_base_data()
    series = df["Sales"].dropna()

    adf_result = adfuller(series)
    kpss_result = kpss(series, regression="c")

    return {
        "adf_statistic": round(float(adf_result[0]), 4),
        "adf_pvalue": round(float(adf_result[1]), 4),
        "kpss_statistic": round(float(kpss_result[0]), 4),
        "kpss_pvalue": round(float(kpss_result[1]), 4),
        "adf_conclusion": "stationary" if adf_result[1] < 0.05 else "non-stationary",
        "kpss_conclusion": "non-stationary" if kpss_result[1] < 0.05 else "stationary",
    }


# -----------------------------------------------------------------------
# TOOL: get_residual_diagnostics
# -----------------------------------------------------------------------
def get_residual_diagnostics(model_name: str) -> dict:
    """
    model_name: one of "arima", "sarima", "prophet", "lstm", "xgboost"
    """
    model_name = model_name.lower()
    df = load_base_data()
    pred = load_forecast(model_name)

    if pred is None:
        return {"error": f"No forecast data found for {model_name}."}

    actual = df["Sales"].tail(len(pred)).values
    n = min(len(actual), len(pred))
    residuals = actual[:n] - pred[:n]

    lb_df = acorr_ljungbox(residuals, lags=[10], return_df=True)
    p_val = float(lb_df["lb_pvalue"].values[0])

    # Flag points that look like anomalies (>2 std from residual mean)
    std = np.std(residuals)
    mean = np.mean(residuals)
    anomaly_idx = np.where(np.abs(residuals - mean) > 2 * std)[0].tolist()

    return {
        "model": model_name.upper(),
        "residual_mean": round(float(mean), 4),
        "residual_std": round(float(std), 4),
        "ljung_box_pvalue": round(p_val, 4),
        "white_noise": bool(p_val > 0.05),
        "num_anomalies": len(anomaly_idx),
        "anomaly_positions": anomaly_idx[:10],  # cap for readability
    }


# -----------------------------------------------------------------------
# TOOL: get_feature_importance (XGBoost)
# -----------------------------------------------------------------------
def get_feature_importance() -> dict:
    """Trains/refits the lightweight XGBoost feature set and returns importances."""
    from xgboost import XGBRegressor

    df = load_base_data()
    temp_df = df[["Date", "Sales"]].copy()
    temp_df["lag1"] = temp_df["Sales"].shift(1)
    temp_df["lag2"] = temp_df["Sales"].shift(2)
    temp_df["lag3"] = temp_df["Sales"].shift(3)
    temp_df["month"] = temp_df["Date"].dt.month
    temp_df["day"] = temp_df["Date"].dt.day
    temp_df["year"] = temp_df["Date"].dt.year
    temp_df = temp_df.dropna()

    features = ["lag1", "lag2", "lag3", "month", "day", "year"]
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(temp_df[features], temp_df["Sales"])

    importances = dict(zip(features, [round(float(x), 4) for x in model.feature_importances_]))
    ranked = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
    return ranked


# -----------------------------------------------------------------------
# TOOL: generate_forecast (XGBoost, adjustable horizon)
# -----------------------------------------------------------------------
def generate_forecast(horizon_days: int = 30) -> dict:
    """
    Trains XGBoost on the full processed dataset and recursively forecasts
    `horizon_days` days into the future. Returns dates + predicted values.
    """
    from xgboost import XGBRegressor

    df = load_base_data()
    temp_df = df[["Date", "Sales"]].copy()
    temp_df["lag1"] = temp_df["Sales"].shift(1)
    temp_df["lag2"] = temp_df["Sales"].shift(2)
    temp_df["lag3"] = temp_df["Sales"].shift(3)
    temp_df["month"] = temp_df["Date"].dt.month
    temp_df["day"] = temp_df["Date"].dt.day
    temp_df["year"] = temp_df["Date"].dt.year
    temp_df = temp_df.dropna()

    features = ["lag1", "lag2", "lag3", "month", "day", "year"]
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(temp_df[features], temp_df["Sales"])

    lag1 = temp_df["Sales"].iloc[-1]
    lag2 = temp_df["Sales"].iloc[-2]
    lag3 = temp_df["Sales"].iloc[-3]
    last_date = temp_df["Date"].iloc[-1]

    forecast, future_dates = [], []
    for _ in range(horizon_days):
        next_date = last_date + pd.Timedelta(days=1)
        input_data = np.array([[lag1, lag2, lag3, next_date.month, next_date.day, next_date.year]])
        pred = float(model.predict(input_data)[0])
        forecast.append(round(pred, 2))
        future_dates.append(str(next_date.date()))
        lag3, lag2, lag1 = lag2, lag1, pred
        last_date = next_date

    return {"dates": future_dates, "predicted_sales": forecast}


# -----------------------------------------------------------------------
# TOOL SCHEMAS (OpenAI-style function schema -- works with Ollama's
# tool-calling models, e.g. llama3.1)
# -----------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_model_metrics",
            "description": "Get RMSE, MSE, MAE performance metrics for all forecasting models (ARIMA, SARIMA, Prophet, LSTM, XGBoost).",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_window": {"type": "integer", "description": "Number of most recent days to evaluate against. Default 30."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stationarity_report",
            "description": "Get ADF and KPSS stationarity test results for the sales time series.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_residual_diagnostics",
            "description": "Get residual analysis, Ljung-Box test, and anomaly flags for a specific model's forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "enum": ["arima", "sarima", "prophet", "lstm", "xgboost"],
                        "description": "Which model's residuals to analyze.",
                    }
                },
                "required": ["model_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature_importance",
            "description": "Get XGBoost feature importance ranking (which features drive sales predictions most).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_forecast",
            "description": "Generate a future sales forecast using XGBoost for a given number of days ahead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon_days": {"type": "integer", "description": "How many days ahead to forecast. Default 30."}
                },
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_model_metrics": get_model_metrics,
    "get_stationarity_report": get_stationarity_report,
    "get_residual_diagnostics": get_residual_diagnostics,
    "get_feature_importance": get_feature_importance,
    "generate_forecast": generate_forecast,
}
