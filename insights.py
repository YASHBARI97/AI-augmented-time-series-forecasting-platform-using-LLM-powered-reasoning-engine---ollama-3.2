"""
ai/insights.py
--------------
One-shot (non-agentic) AI narrative generators. These take structured
output already computed by ai/tools.py and turn it into plain-English
text for the dashboard — model comparison summaries, feature-importance
explanations, anomaly hypotheses, and a full executive report.

Kept separate from ai/chat.py because these don't need tool-calling:
the data is already fetched, we just need Claude to narrate it.
"""

import json

from ai.client import simple_completion, get_fast_model, get_reasoning_model
from ai import tools


def model_comparison_narrative() -> str:
    metrics = tools.get_model_metrics()
    prompt = f"""Here are forecasting model performance metrics (lower is better):

{json.dumps(metrics, indent=2)}

Write a short (3-5 sentence) plain-English summary for a business audience:
which model performs best and by roughly how much, and one plausible reason
why (e.g. handles non-linearity, captures seasonality, needs more data).
Do not invent numbers not shown above."""
    return simple_completion(prompt, model=get_fast_model())


def feature_importance_narrative() -> str:
    importances = tools.get_feature_importance()
    prompt = f"""XGBoost feature importances (higher = more influential on sales predictions):

{json.dumps(importances, indent=2)}

Explain in 3-4 plain-English sentences which factors drive the sales forecast
most, and what that implies for the business (e.g. planning around recent
trend vs. seasonality)."""
    return simple_completion(prompt, model=get_fast_model())


def anomaly_narrative(model_name: str) -> str:
    diagnostics = tools.get_residual_diagnostics(model_name)
    if "error" in diagnostics:
        return diagnostics["error"]

    prompt = f"""Residual diagnostics for the {model_name.upper()} model:

{json.dumps(diagnostics, indent=2)}

In 3-4 sentences, explain in plain English what this means for forecast
reliability. If num_anomalies > 0, offer 1-2 *plausible, clearly-labeled
hypotheses* for what might cause outlier errors (e.g. promotions, holidays,
data gaps, demand shocks) — do not state them as confirmed facts."""
    return simple_completion(prompt, model=get_fast_model())


def executive_report() -> str:
    """
    Full markdown report combining stationarity, model comparison, best
    model diagnostics, and a forward-looking business recommendation.
    Used by the 'Generate Report' button in the Streamlit app.
    """
    stationarity = tools.get_stationarity_report()
    metrics = tools.get_model_metrics()
    importances = tools.get_feature_importance()

    # Determine best model by RMSE (skip None values)
    valid = [m for m in metrics if m["rmse"] is not None]
    best_model = min(valid, key=lambda m: m["rmse"])["model"] if valid else "N/A"
    diagnostics = tools.get_residual_diagnostics(best_model.lower()) if best_model != "N/A" else {}

    prompt = f"""Write a concise executive report (markdown, ~350-450 words) for a
sales forecasting project, using ONLY the data below. Structure it with these
headers: ## Summary, ## Model Performance, ## Key Drivers, ## Reliability &
Risks, ## Business Recommendations.

Stationarity tests:
{json.dumps(stationarity, indent=2)}

Model comparison metrics:
{json.dumps(metrics, indent=2)}

Best performing model: {best_model}

Best model residual diagnostics:
{json.dumps(diagnostics, indent=2)}

XGBoost feature importances:
{json.dumps(importances, indent=2)}

Keep it business-readable, avoid inventing numbers not given above, and
clearly label any causal hypotheses as hypotheses."""
    return simple_completion(prompt, model=get_reasoning_model(), max_tokens=1500)
