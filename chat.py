"""
ai/chat.py
----------
Implements the "Chat with your Forecast" agent using a local Ollama model
(free, no API key). Automatically detects whichever free local model you
have installed (see ai/client.py).

If the installed model doesn't support tool-calling well (common with
smaller models), this gracefully falls back to a plain-answer mode instead
of crashing, so AI Chat always returns *something* useful.
"""

import json

import ollama

from ai.client import get_reasoning_model
from ai.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

SYSTEM_PROMPT = """You are an AI analyst embedded in a sales/demand forecasting dashboard.
You have tools to query model performance metrics, stationarity tests, residual
diagnostics, feature importance, and to generate new forecasts.

Rules:
- Only answer questions related to this forecasting project's data and models.
- Always use tools to get real numbers before making claims -- never guess metric values.
- Keep answers concise, business-friendly, and avoid raw jargon unless asked.
- When you flag an anomaly or trend cause, clearly label it as a hypothesis, not fact.
- If a question is unrelated to the forecasting project, politely redirect.
"""

MAX_TOOL_ROUNDS = 5


def _plain_fallback(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    Used when the model doesn't support tool-calling. Instead of calling
    tools dynamically, we just grab the most commonly useful data
    (model metrics) and hand it to the model as context.
    """
    from ai import tools as pipeline_tools

    context = {
        "model_metrics": pipeline_tools.get_model_metrics(),
        "stationarity": pipeline_tools.get_stationarity_report(),
    }
    messages = history + [
        {
            "role": "user",
            "content": (
                f"{user_message}\n\n"
                f"(Reference data you can use to answer, if relevant:\n"
                f"{json.dumps(context, default=str)})"
            ),
        }
    ]
    response = ollama.chat(model=get_reasoning_model(), messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages)
    answer = response["message"].get("content", "")
    messages.append({"role": "assistant", "content": answer})
    return answer, messages


def run_chat_turn(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """
    Runs one full chat turn (including any tool-call rounds).
    Returns (final_text_answer, updated_history).
    """
    if not history:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = history + [{"role": "user", "content": user_message}]
    model = get_reasoning_model()

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = ollama.chat(model=model, messages=messages, tools=TOOL_SCHEMAS)
            msg = response["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return msg.get("content", ""), messages

            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments", {}) or {}
                func = TOOL_FUNCTIONS.get(name)
                try:
                    result = func(**args) if func else {"error": f"Unknown tool {name}"}
                except Exception as e:
                    result = {"error": str(e)}

                messages.append({"role": "tool", "content": json.dumps(result, default=str)})

        return "I wasn't able to finish reasoning about that in time -- try a more specific question.", messages

    except Exception:
        # Model doesn't support tool-calling, or Ollama rejected the tools
        # parameter -- fall back to a plain context-stuffed answer instead
        # of showing an error screen.
        return _plain_fallback(user_message, history)
