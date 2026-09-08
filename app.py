import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ai.client import is_configured, setup_instructions
from ai import insights, chat as ai_chat


st.set_page_config(layout="wide", page_title="Sales Forecasting Dashboard")
st.title("📊 AI-augmented time-series forecasting platform using LLM-powered reasoning engine - ollama 3.2")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data

def load_base_data():
    df = pd.read_csv("data/processed_data.csv")
    
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_base_data()

# -----------------------------
# LOAD FORECAST FUNCTION
# -----------------------------
def load_forecast(name):
    try:
        data = np.load(f"data/{name}_forecast.npy")
        return data.flatten()
    except:
        return None

# Load all forecasts
arima = load_forecast("arima")
sarima = load_forecast("sarima")
prophet = load_forecast("prophet")
lstm = load_forecast("lstm")
xgboost = load_forecast("xgboost")

# -----------------------------
# SIDEBAR
# -----------------------------
section = st.sidebar.radio(
    "Select Section",
    ["Overview", "Stationarity", "Models", "Evaluation", "Diagnostics", "Future Forecast",
     "🤖 AI Insights", "💬 AI Chat", "📄 AI Report"]
)

if not is_configured():
    st.sidebar.warning(setup_instructions(), icon="⚠️")

# -----------------------------
# OVERVIEW
# -----------------------------
if section == "Overview":
    # 1. NEONIC CSS INJECTION (Deep Royal Blue Theme)
    st.markdown("""
        <style>
        /* Card-like containers for metrics */
        [data-testid="stMetric"] {
            background-color: rgba(0, 0, 255, 0.05); /* Very light blue tint */
            padding: 15px;
            border-radius: 15px;
            border: 1px solid rgba(0, 0, 255, 0.4); /* Matches your image blue */
            box-shadow: 0 0 15px rgba(0, 0, 255, 0.15);
        }
        /* Neon Glow for Metric Values */
        [data-testid="stMetricValue"] {
            color: #0000FF !important; /* Pure Blue to match graph */
            text-shadow: 0 0 8px rgba(0, 0, 255, 0.4); 
            font-family: 'Courier New', Courier, monospace;
        }
        /* Style the Expander */
        .streamlit-expanderHeader {
            font-weight: bold;
            color: #0000FF !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # 2. KEY PERFORMANCE INDICATORS
    st.markdown("### 💎 KPI's Metrics")
    col1, col2, col3 = st.columns(3)
    
    total_sales = int(df['Sales'].sum())
    total_revenue = int(df['Revenue'].sum())
    total_profit = int(df['Profit'].sum())

    col1.metric("TOTAL SALES", f"{total_sales:,}")
    col2.metric("TOTAL REVENUE", f"₹ {total_revenue:,}")
    col3.metric("TOTAL PROFIT", f"₹ {total_profit:,}")

    st.divider()

    # 3. HISTORICAL TREND (Cinematic Chart)
    st.markdown("### 📈 Historical Sales Trend")
    # Using container width for cinematic feel
    st.line_chart(df.set_index('Date')['Sales'], color="#0011FF")

    # 4. DATA VIEW (In expander to keep it clean)
    with st.expander("Data Preview"):
        st.subheader("Optimized Dataset Preview")
        
        # We only apply .2f to float columns. Category and Product are STRINGS.
        # Removing Category and Product from format subset avoids errors.
        styled_df = df.head(50).style.format({
            'Sales': "{:}",       # Whole number with commas
            'Revenue': "{:,.2f}",  # 2 decimal places with commas
            'Profit': "{:,.2f}",
            'Cost': "{:,.2f}"
        })
        
        st.dataframe(styled_df, use_container_width=True)
# -----------------------------
# STATIONARITY
# -----------------------------
elif section == "Stationarity":
    st.subheader("📊 Stationarity Test Results")
    
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # 1. RUN TESTS DIRECTLY
    series = df['Sales'].dropna()
    
    with st.spinner('Calculating statistical tests...'):
        adf_result = adfuller(series)
        kpss_result = kpss(series, regression='c')

        # 2. CREATE THE MATRIX
        results_dict = {
            "ADF Statistic": round(adf_result[0], 4),
            "ADF p-value": round(adf_result[1], 4),
            "KPSS Statistic": round(kpss_result[0], 4),
            "KPSS p-value": round(kpss_result[1], 4)
        }
        
        df_matrix = pd.DataFrame(list(results_dict.items()), columns=["Metric", "Value"])
        st.table(df_matrix)

    # 3. INTERPRETATION
    st.subheader("📌 Statistical Interpretation")
    c1, c2 = st.columns(2)
    with c1:
        if results_dict["ADF p-value"] < 0.05:
            st.success(f"ADF: Stationary (p={results_dict['ADF p-value']})")
        else:
            st.error("ADF: Non-Stationary")
    with c2:
        if results_dict["KPSS p-value"] < 0.05:
            st.error("KPSS: Non-Stationary")
        else:
            st.success(f"KPSS: Stationary (p={results_dict['KPSS p-value']})")

    # 4. VISUAL DIAGNOSTICS
    st.divider()
    st.subheader("📉 Visual Diagnostics (ACF & PACF)")
    
    v1, v2 = st.columns(2)
    with v1:
        st.write("**Autocorrelation (ACF)**")
        fig1, ax1 = plt.subplots()
        plot_acf(series, ax=ax1, lags=40)
        st.pyplot(fig1)

    with v2:
        st.write("**Partial Autocorrelation (PACF)**")
        fig2, ax2 = plt.subplots()
        plot_pacf(series, ax=ax2, lags=40)
        st.pyplot(fig2)
                
# -----------------------------
# MODELS
# -----------------------------
elif section == "Models":
    model_choice = st.selectbox(
        "Select Model to Visualize",
        ["ARIMA", "SARIMA", "Prophet", "LSTM", "XGBoost", "Comparison"]
    )

    def plot_model(name, forecast_data):
        if forecast_data is None:
            st.warning(f"No forecast data found for {name}. Run the {name.lower()}.py script first.")
            return

        fig, ax = plt.subplots(figsize=(10, 4))
        recent_df = df.tail(100)
        ax.plot(recent_df['Date'], recent_df['Sales'], label="Actual (Recent)", color='black', alpha=0.6)

        forecast_len = len(forecast_data)
        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(last_date, periods=forecast_len + 1, freq='D')[1:]

        ax.plot(future_dates, forecast_data, label=f"Predicted {name}", linewidth=2)
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if model_choice == "Comparison":
        st.subheader("📊 Multi-Model Comparison")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Date'].tail(50), df['Sales'].tail(50), label="Actual", color='black')
        
        models_to_compare = {"ARIMA": arima, "SARIMA": sarima, "Prophet": prophet, "LSTM": lstm, "XGBoost": xgboost}
        last_date = df['Date'].iloc[-1]
        
        for m_name, m_data in models_to_compare.items():
            if m_data is not None:
                f_dates = pd.date_range(last_date, periods=len(m_data) + 1, freq='D')[1:]
                ax.plot(f_dates, m_data, label=m_name)
        
        ax.legend()
        st.pyplot(fig)
    else:
        data_map = {"ARIMA": arima, "SARIMA": sarima, "Prophet": prophet, "LSTM": lstm, "XGBoost": xgboost}
        st.subheader(f"{model_choice} Prediction")
        plot_model(model_choice, data_map[model_choice])

# -----------------------------
# EVALUATION
# -----------------------------
elif section == "Evaluation":
    st.subheader("📊 Model Performance Metrics")
    actual_test = df['Sales'].tail(30).values 

    def get_metrics(pred, actual):
        if pred is None: return ["N/A"] * 3
        n = min(len(pred), len(actual))
        p, a = pred[:n], actual[:n]
        return [
            f"{mean_squared_error(a, p):.2f}",
            f"{np.sqrt(mean_squared_error(a, p)):.2f}",
            f"{mean_absolute_error(a, p):.2f}"
        ]

    metrics_data = [
        ["ARIMA"] + get_metrics(arima, actual_test),
        ["SARIMA"] + get_metrics(sarima, actual_test),
        ["Prophet"] + get_metrics(prophet, actual_test),
        ["LSTM"] + get_metrics(lstm, actual_test),
        ["XGBoost"] + get_metrics(xgboost, actual_test),
    ]

    results_df = pd.DataFrame(metrics_data, columns=["Model", "MSE", "RMSE", "MAE"])
    st.table(results_df)

# -----------------------------
# DIAGNOSTICS (Residuals & Ljung-Box)
# -----------------------------
elif section == "Diagnostics":
    st.subheader("🛠️ Model Diagnostics: Residual Analysis & Ljung-Box Test")
    
    diag_choice = st.selectbox("Select Model to Diagnose", ["ARIMA", "SARIMA", "Prophet", "LSTM", "XGBoost"])
    
    # Map selection to loaded data
    data_map = {"ARIMA": arima, "SARIMA": sarima, "Prophet": prophet, "LSTM": lstm, "XGBoost": xgboost}
    pred = data_map[diag_choice]
    
    # We use the end of the processed data as the 'actual' comparison
    actual = df['Sales'].tail(len(pred) if pred is not None else 30).values

    if pred is not None:
        n = min(len(actual), len(pred))
        residuals = actual[:n] - pred[:n]

        # --- Residual Plot ---
        st.write(f"### {diag_choice} Residuals Plot")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(residuals, color='teal', label='Residuals (Actual - Predicted)')
        ax.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax.set_title(f"Residual Errors for {diag_choice}")
        ax.legend()
        st.pyplot(fig)

        # --- Ljung-Box Test ---
        st.write("### Ljung-Box Test for Autocorrelation")
        lb_df = acorr_ljungbox(residuals, lags=[10], return_df=True)
        p_val = lb_df['lb_pvalue'].values[0]
        
        c1, c2 = st.columns(2)
        c1.metric("Mean of Residuals", f"{np.mean(residuals):.4f}")
        c2.metric("Ljung-Box p-value (Lag 10)", f"{p_val:.4f}")

        if p_val > 0.05:
            st.success("✅ Results: Residuals are independent (White Noise). The model has captured the patterns well.")
        else:
            st.error("❌ Results: Residuals are correlated. The model might be missing some temporal patterns.")
    else:
        st.error(f"Data for {diag_choice} not found. Please run the model script first.")

# -----------------------------
# FUTURE FORECAST
# -----------------------------

elif section == "Future Forecast":

    st.subheader("🚀 Adaptive Model Forecast (XGBoost)")

    # -----------------------------
    # DROPDOWN (Dynamic Horizon)
    # -----------------------------
    horizon_options = {
        "Next 1 Month (30 Days)": 30,
        "Next 3 Months (90 Days)": 90,
        "Next 1 Year (365 Days)": 365
    }

    selected_option = st.selectbox("Select Forecast Horizon", list(horizon_options.keys()))
    future_days = horizon_options[selected_option]

    # -----------------------------
    # BUTTON TRIGGER
    # -----------------------------
    if st.button("Forecast"):

        with st.spinner("Generating Forecast..."):

            try:
                from xgboost import XGBRegressor

                # -----------------------------
                # DATA PREPARATION
                # -----------------------------
                temp_df = df[['Date', 'Sales']].copy()

                temp_df['lag1'] = temp_df['Sales'].shift(1)
                temp_df['lag2'] = temp_df['Sales'].shift(2)
                temp_df['lag3'] = temp_df['Sales'].shift(3)

                temp_df['month'] = temp_df['Date'].dt.month
                temp_df['day'] = temp_df['Date'].dt.day
                temp_df['year'] = temp_df['Date'].dt.year

                temp_df = temp_df.dropna()

                features = ['lag1', 'lag2', 'lag3', 'month', 'day', 'year']

                # -----------------------------
                # MODEL TRAINING
                # -----------------------------
                model = XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5
                )

                model.fit(temp_df[features], temp_df['Sales'])

                # -----------------------------
                # RECURSIVE FORECAST
                # -----------------------------
                lag1 = temp_df['Sales'].iloc[-1]
                lag2 = temp_df['Sales'].iloc[-2]
                lag3 = temp_df['Sales'].iloc[-3]
                
                last_date = temp_df['Date'].iloc[-1]
           

                forecast = []
                future_dates = []

                for _ in range(future_days):

                    next_date = last_date + pd.Timedelta(days=1)

                    input_data = np.array([[
                        lag1, lag2, lag3,
                        next_date.month, next_date.day, next_date.year
                    ]])

                    pred = model.predict(input_data)[0]

                    forecast.append(pred)
                    future_dates.append(next_date)

                    # update lags
                    lag3 = lag2
                    lag2 = lag1
                    lag1 = pred

                    last_date = next_date

                # -----------------------------
                # RESULT
                # -----------------------------
                result_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted_Sales": forecast,
                }).set_index("Date")

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.write("📄 Forecast Data")
                    st.dataframe(result_df)

                with col2:
                    st.write("📈 Forecast Visualization")
                    st.line_chart(result_df[['Predicted_Sales']])

                st.success(f"✅ Forecast generated for {future_days} days")

            except Exception as e:
                st.error(f"❌ Error: {e}")

# -----------------------------
# AI INSIGHTS
# -----------------------------
elif section == "🤖 AI Insights":
    st.subheader("🤖 AI-Generated Insights")
    st.caption("Claude reads your model metrics, diagnostics, and feature importances, and narrates them in plain English.")

    if not is_configured():
        st.error(setup_instructions())
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🔑 Key Drivers", "⚠️ Anomaly Check"])

        with tab1:
            if st.button("Generate model comparison summary"):
                with st.spinner("Analyzing model metrics..."):
                    st.write(insights.model_comparison_narrative())

        with tab2:
            if st.button("Explain feature importance"):
                with st.spinner("Analyzing XGBoost feature importances..."):
                    st.write(insights.feature_importance_narrative())

        with tab3:
            diag_model = st.selectbox("Model to check for anomalies", ["ARIMA", "SARIMA", "Prophet", "LSTM", "XGBoost"], key="anomaly_model")
            if st.button("Run anomaly check"):
                with st.spinner(f"Analyzing {diag_model} residuals..."):
                    st.write(insights.anomaly_narrative(diag_model.lower()))

# -----------------------------
# AI CHAT
# -----------------------------
elif section == "💬 AI Chat":
    st.subheader("💬 Chat With Your Forecast")
    st.caption("Ask about model performance, drivers, diagnostics, or request a fresh forecast — Claude will call the right tool.")

    if not is_configured():
        st.error(setup_instructions())
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []       # Anthropic-format messages (for the API)
        if "chat_display" not in st.session_state:
            st.session_state.chat_display = []        # (role, text) pairs (for rendering)

        for role, text in st.session_state.chat_display:
            with st.chat_message(role):
                st.markdown(text)

        example_qs = [
            "Which model performs best and why?",
            "Are the XGBoost residuals reliable?",
            "Forecast the next 14 days",
        ]
        cols = st.columns(len(example_qs))
        clicked = None
        for c, q in zip(cols, example_qs):
            if c.button(q):
                clicked = q

        user_input = st.chat_input("Ask about your forecast...") or clicked

        if user_input:
            st.session_state.chat_display.append(("user", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, updated_history = ai_chat.run_chat_turn(user_input, st.session_state.chat_history)
                    st.markdown(answer)

            st.session_state.chat_history = updated_history
            st.session_state.chat_display.append(("assistant", answer))

        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.session_state.chat_display = []
            st.rerun()

# -----------------------------
# AI REPORT
# -----------------------------
elif section == "📄 AI Report":
    st.subheader("📄 AI-Generated Executive Report")
    st.caption("Combines stationarity tests, model comparison, best-model diagnostics, and feature drivers into one business-readable report.")

    if not is_configured():
        st.error(setup_instructions())
    else:
        if st.button("Generate Report"):
            with st.spinner("Compiling data and writing report..."):
                report_md = insights.executive_report()
                st.session_state["last_report"] = report_md

        if "last_report" in st.session_state:
            st.markdown(st.session_state["last_report"])
            st.download_button(
                "⬇️ Download Report (Markdown)",
                data=st.session_state["last_report"],
                file_name="forecast_executive_report.md",
                mime="text/markdown",
            )