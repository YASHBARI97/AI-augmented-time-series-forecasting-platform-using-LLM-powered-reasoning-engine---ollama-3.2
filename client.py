"""
ai/client.py
------------
Free, local AI client using Ollama (https://ollama.com) -- no API key,
no billing, no internet required after setup.

This version AUTO-DETECTS whichever model you have pulled locally, so it
works regardless of which model name you installed. It deliberately
ignores ":cloud" models (e.g. "glm-5.3-flash:cloud") since those require
a paid Ollama plan -- only genuinely free, fully-local models are used.
"""

import ollama

_cached_model = None

# If you have multiple models installed, these are tried in order of
# preference (best tool-calling / quality first).
_PREFERRED_MODELS = ["llama3.1", "llama3.2", "llama3", "qwen2.5", "mistral"]


def _list_local_models() -> list[str]:
    """Returns installed model names, excluding paid ':cloud' models."""
    try:
        raw = ollama.list().get("models", [])
    except Exception:
        return []

    names = []
    for m in raw:
        name = m.get("model") or m.get("name")
        if name and ":cloud" not in name:
            names.append(name)
    return names


def _pick_local_model() -> str | None:
    """Finds the best available free local model, caching the result."""
    global _cached_model
    if _cached_model:
        return _cached_model

    local_models = _list_local_models()
    if not local_models:
        return None

    for preferred in _PREFERRED_MODELS:
        for name in local_models:
            if name.startswith(preferred):
                _cached_model = name
                return name

    # No preferred match -- just use whatever free local model exists
    _cached_model = local_models[0]
    return local_models[0]


def get_reasoning_model() -> str:
    """Model used for the chat agent and executive report."""
    return _pick_local_model() or "llama3.2"


def get_fast_model() -> str:
    """Model used for quick one-shot narratives."""
    return _pick_local_model() or "llama3.2"


def is_configured() -> bool:
    """True if Ollama is running AND at least one free local model is pulled."""
    return _pick_local_model() is not None


def setup_instructions() -> str:
    return (
        "No free local model found. Run this in a terminal:\n\n"
        "    ollama pull llama3.2\n\n"
        "(about 2GB, works on most laptops) then refresh this page."
    )


def simple_completion(prompt: str, system: str = "", model: str | None = None, max_tokens: int = 800) -> str:
    """One-shot text completion, no tools. Used for narratives/summaries."""
    model = model or get_fast_model()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(model=model, messages=messages, options={"num_predict": max_tokens})
    return response["message"]["content"]
