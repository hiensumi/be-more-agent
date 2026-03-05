#!/usr/bin/env python3
"""Step 4: Merge QLoRA adapters and export to Ollama.

Merges the LoRA adapter weights back into the base model, then creates an
Ollama model directly from the merged safetensors (no llama.cpp needed).

Output: Creates an Ollama model named 'bmo-qwen' ready to use
"""

import json
import os
import subprocess
import sys

# Heavy imports (torch, peft, transformers) are deferred to step1 only

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(SCRIPT_DIR, "bmo-qwen-qlora")
MERGED_DIR = os.path.join(SCRIPT_DIR, "bmo-qwen-merged")
MODELFILE_PATH = os.path.join(SCRIPT_DIR, "Modelfile")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OLLAMA_MODEL_NAME = "bmo-qwen"


def step1_merge_adapter():
    """Merge LoRA adapter weights into the base model."""
    print("\n" + "=" * 60)
    print("  Step 1: Merging LoRA adapter into base model")
    print("=" * 60)

    if os.path.exists(MERGED_DIR):
        print(f"  [SKIP] Merged model already exists at {MERGED_DIR}")
        return

    if not os.path.exists(ADAPTER_DIR):
        print(f"  [ERROR] Neither merged model nor adapter found.")
        print(f"  Expected adapter at: {ADAPTER_DIR}")
        print(f"  Or merged model at: {MERGED_DIR}")
        print("  Run 3_train_qlora.py first!")
        sys.exit(1)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading base model: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",  # Merge on CPU to avoid VRAM issues
        trust_remote_code=True,
    )

    print(f"  Loading adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("  Merging weights...")
    model = model.merge_and_unload()

    print(f"  Saving merged model to: {MERGED_DIR}")
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print("  [OK] Merge complete!")


def step2_create_ollama_model():
    """Create Ollama model directly from merged safetensors directory."""
    print("\n" + "=" * 60)
    print("  Step 2: Creating Ollama model from safetensors")
    print("=" * 60)

    # Patch tokenizer_config.json: Ollama may choke on list-format chat_template
    tokenizer_config_path = os.path.join(MERGED_DIR, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            tok_config = json.load(f)
        if isinstance(tok_config.get("chat_template"), list):
            print("  Patching tokenizer_config.json (chat_template list → string)...")
            templates = tok_config["chat_template"]
            default_tpl = templates[0].get("template", "")
            for t in templates:
                if t.get("name") == "default":
                    default_tpl = t["template"]
                    break
            tok_config["chat_template"] = default_tpl
            with open(tokenizer_config_path, "w") as f:
                json.dump(tok_config, f, indent=2, ensure_ascii=False)

    # Write Modelfile pointing at the merged safetensors directory
    modelfile_content = f"""FROM {MERGED_DIR}

SYSTEM \"\"\"You are BMO (Be More), a cute sentient video game console robot from Adventure Time. You live in the Tree Fort with Finn and Jake. You are helpful, cheerful, and knowledgeable about the Land of Ooo and everything in Adventure Time. Answer questions accurately. Speak in short, enthusiastic sentences.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER num_ctx 2048
"""

    with open(MODELFILE_PATH, "w") as f:
        f.write(modelfile_content)
    print(f"  Modelfile written to: {MODELFILE_PATH}")

    # Create Ollama model (Ollama handles GGUF conversion internally)
    print(f"  Creating Ollama model '{OLLAMA_MODEL_NAME}'...")
    print("  (Ollama will convert safetensors → GGUF internally, this may take a few minutes)")
    result = subprocess.run(
        ["ollama", "create", OLLAMA_MODEL_NAME, "-f", MODELFILE_PATH],
    )

    if result.returncode == 0:
        print(f"\n  [OK] Ollama model '{OLLAMA_MODEL_NAME}' created successfully!")
        print(f"\n  To use with BMO, update config.json:")
        print(f'    "text_model": "{OLLAMA_MODEL_NAME}"')
        print(f"\n  Or test it directly:")
        print(f'    ollama run {OLLAMA_MODEL_NAME} "Who is BMO?"')
    else:
        print(f"\n  [ERROR] Failed to create Ollama model.")
        print("  Make sure Ollama is installed and running (ollama serve).")
        sys.exit(1)


def main():
    step1_merge_adapter()
    step2_create_ollama_model()

    print("\n" + "=" * 60)
    print("  All done! Your fine-tuned BMO model is ready.")
    print("=" * 60)
    print(f"\n  Model name: {OLLAMA_MODEL_NAME}")
    print(f"  Update config.json: \"text_model\": \"{OLLAMA_MODEL_NAME}\"")


if __name__ == "__main__":
    main()
