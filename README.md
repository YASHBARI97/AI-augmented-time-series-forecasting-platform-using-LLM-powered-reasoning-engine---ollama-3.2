AI-augmented time-series forecasting platform using LLM-powered reasoning engine - ollama 3.2

Project Overview

This project analyzes and forecasts hourly energy demand using statistical and machine learning–based time series models, then layers an AI reasoning engine on top so the results can be queried, explained, and reported on in plain English through an interactive web dashboard.

The study follows a systematic approach: data preprocessing, stationarity testing, model development and benchmarking across five forecasting approaches, residual diagnostics, evaluation using multiple performance metrics, and — new in this version — an integrated AI layer that generates explainable insights, answers natural-language questions about the models, and produces decision-ready reports.

📂 Dataset Information
Source: NITI Aayog – Government of India
Description: Hourly electricity demand readings from national energy records.
Purpose: Understand temporal variations in energy consumption and forecast future demand.
Frequency: Hourly
Data Format: Time-indexed CSV/Excel data
🧪 Statistical Tests Performed

1. Augmented Dickey-Fuller (ADF) Test

Checks stationarity in time series data.
H₀: Data is non-stationary | H₁: Data is stationary
Performed before and after differencing to ensure stationarity.

2. Kwiatkowski–Phillips–Schmidt–Shin (KPSS) Test

Complements ADF by testing the opposite hypothesis.
H₀: Data is stationary | H₁: Data is non-stationary

3. Ljung–Box Test

Checks autocorrelation in residuals after model fitting.
If p-value > 0.05 → residuals are independent (no autocorrelation).
🔍 Analytical Steps

Residual Analysis — conducted after model fitting to ensure white noise behavior:

Checked residual mean and variance stability
Plotted residual distribution
Performed Ljung–Box test to confirm randomness

Time Series Decomposition — decomposed data into trend, seasonal, and residual components to visualize long-term patterns.

📈 Visualization & Plots
Plot Type	Description
ACF Plot (Autocorrelation Function)	Identifies lag correlations (for AR terms)
PACF Plot (Partial Autocorrelation Function)	Helps determine MA terms
Actual vs Forecast Plot	Compares actual vs predicted energy demand
Residual Forecast Plot	Shows randomness and residual behavior
Predicted Forecast Plot	Displays the future forecast curve
Feature Importance Plot	Ranks the drivers behind XGBoost's predictions

These visualizations guided model selection and evaluation, and are rendered across 20+ charts in the Streamlit dashboard.

🤖 Models Implemented

1. ARIMA (AutoRegressive Integrated Moving Average)

Baseline forecasting model for univariate time series.
Parameters (p, d, q) tuned via ACF/PACF analysis.
Evaluated using AIC/BIC and forecast error metrics.

2. SARIMA (Seasonal ARIMA)

Handles both trend and seasonality.
Seasonal parameters (P, D, Q, m) optimized using AIC/BIC.
Demonstrated better accuracy for cyclical energy demand.

3. Prophet (Meta/Facebook)

Captures complex patterns automatically (trend + seasonality).
Produces interpretable and visually rich forecasts.

4. LSTM (Long Short-Term Memory)

Deep learning model capturing long-range temporal dependencies.
Benchmarked alongside classical statistical models for comparison.

5. XGBoost

Gradient-boosted tree model using lag and calendar features.
Outperformed all other models, delivering the lowest RMSE, MSE, and MAE across forecasting horizons.
📊 Model Evaluation Metrics
Metric	Description
AIC	Model fit quality (lower = better)
BIC	Penalizes overly complex models (lower = better)
RMSE	Measures average magnitude of error
MSE	Average squared prediction error
MAE / MAPE	Accuracy in absolute/percentage terms (lower = better)

XGBoost delivered the highest overall forecasting accuracy, with SARIMA and Prophet performing best among the classical/statistical approaches.

🤖 AI-Powered Layer (New)

The project now includes an integrated AI reasoning layer, built with a locally-hosted LLM via Ollama (free, no API costs), that sits on top of the forecasting pipeline:

AI Insights — auto-generates plain-English narratives explaining model comparison results, XGBoost feature importance, and residual-based anomaly hypotheses, removing the need to manually interpret every metric.
AI Chat — a natural-language interface where users can ask questions like "Which model performs best and why?" or "Forecast the next 14 days"; the AI uses tool-calling to query the actual pipeline (metrics, diagnostics, forecasts) rather than guessing answers.
AI Report — one-click generation of a decision-ready executive report combining stationarity results, model benchmarks, key drivers, and business recommendations, exportable as a markdown document.

This turns the project from a static analysis notebook into an interactive, explainable, AI-augmented forecasting product.

⚙️ Workflow Summary
Data import from NITI Aayog
Data cleaning and indexing
Visualization of hourly demand
Stationarity tests (ADF, KPSS)
ACF/PACF analysis
Model building (ARIMA, SARIMA, Prophet, LSTM, XGBoost)
Model diagnostics & residual analysis
Forecasting & evaluation (AIC, BIC, RMSE, MAE)
AI layer integration — insight generation, natural-language chat, automated reporting
Deployment via interactive Streamlit web dashboard
🚀 Key Insights
Strong daily and weekly seasonality detected in energy demand.
SARIMA outperformed ARIMA for short-term forecasts; Prophet captured long-term seasonality well.
XGBoost outperformed all models overall, benefiting from lag and calendar-based feature engineering.
Residuals confirmed model validity and randomness across all approaches.
The AI layer surfaces these findings conversationally and in auto-generated reports, making the forecasting system usable by non-technical stakeholders — not just data scientists.
