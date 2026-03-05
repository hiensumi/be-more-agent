import json
import os

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
    "system_prompt_extras": ""
}

OLLAMA_OPTIONS = {
    "keep_alive": "-1",
    "num_thread": 4,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
}

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
