"""Intent classifier using FastText — decides if user input needs a web search.

Uses a lightweight FastText model (~414KB) trained on labeled examples.
Classifies in microseconds with high confidence.

Labels:
  search → auto-fetch from DuckDuckGo before answering
  chat   → let the main LLM handle (chat or action)
"""

import os
import warnings

import fasttext

warnings.filterwarnings("ignore", category=UserWarning, module="fasttext")

# ── Model path ───────────────────────────────────────────────────────────────

_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "classifier_model.bin")
_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            print(f"[CLASSIFIER] Model not found at {_MODEL_PATH}", flush=True)
            return None
        _model = fasttext.load_model(_MODEL_PATH)
        print(f"[CLASSIFIER] FastText model loaded ({os.path.getsize(_MODEL_PATH) / 1024:.0f} KB)", flush=True)
    return _model


def classify_input(text: str) -> str:
    """Classify user input using FastText.

    Returns
    -------
    str
        'search' – auto-search recommended
        'chat'   – let the main LLM handle (chat or action)
    """
    stripped = text.strip()
    if not stripped:
        return "chat"

    model = _get_model()
    if model is None:
        return "chat"

    try:
        labels, probs = model.predict(stripped)
        label = labels[0].replace("__label__", "")
        confidence = probs[0]
        return label if confidence > 0.5 else "chat"
    except Exception as e:
        print(f"[CLASSIFIER] Error: {e}", flush=True)
        return "chat"
