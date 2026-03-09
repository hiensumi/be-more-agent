import json
import os

import ollama

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"
STATIC_MEMORY_FILE = "static_memory.json"
BMO_IMAGE_FILE = "current_image.jpg"
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.5

INPUT_DEVICE_NAME = None

DEFAULT_CONFIG = {
    "text_model": "qwen2.5:3b",
    "vision_model": "moondream",
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "chat_memory": False,
    "camera_rotation": 0,
    "system_prompt_extras": "",
    "base_prompt": "default",
    "ollama_host": "",
    "tavily_api_key": "",
    "game_path": ""
}

OLLAMA_OPTIONS = {
    "keep_alive": "-1",
    "num_thread": 4,
    "num_ctx": 2048,
    "num_predict": 200,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
}

def get_dynamic_options(messages=None):
    """Return OLLAMA_OPTIONS with num_ctx scaled to the conversation size."""
    opts = OLLAMA_OPTIONS.copy()
    if messages:
        # Rough estimate: ~4 chars per token
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4
        # Add headroom for the response
        needed = estimated_tokens + opts["num_predict"] + 256
        # Clamp between 2048 and 8192
        opts["num_ctx"] = max(2048, min(needed, 8192))
    return opts

WEB_PORT = 8585

greeting_sounds_dir = "sounds/greeting_sounds"
ack_sounds_dir = "sounds/ack_sounds"
thinking_sounds_dir = "sounds/thinking_sounds"
error_sounds_dir = "sounds/error_sounds"


def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            print(f"Config Error: {e}. Using defaults.")
    return config


CURRENT_CONFIG = load_config()
TEXT_MODEL = CURRENT_CONFIG["text_model"]
VISION_MODEL = CURRENT_CONFIG["vision_model"]

_ollama_host = CURRENT_CONFIG.get("ollama_host", "").strip()
if _ollama_host:
    OLLAMA_CLIENT = ollama.Client(host=_ollama_host)
    print(f"[CONFIG] Using remote Ollama: {_ollama_host}", flush=True)
else:
    OLLAMA_CLIENT = ollama.Client()
    print("[CONFIG] Using local Ollama", flush=True)
