#!/usr/bin/env python3
"""Step 4: Merge QLoRA adapters and export to Ollama.

Merges the LoRA adapter weights back into the base model, converts to GGUF
format, and creates an Ollama model ready to use with BMO.

Requires: llama-cpp-python (for GGUF conversion) OR llama.cpp built from source

Output: Creates an Ollama model named 'bmo-qwen' ready to use
"""

import os
import shutil
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(SCRIPT_DIR, "bmo-qwen-qlora")
MERGED_DIR = os.path.join(SCRIPT_DIR, "bmo-qwen-merged")
GGUF_DIR = os.path.join(SCRIPT_DIR, "bmo-qwen-gguf")
GGUF_FILE = os.path.join(GGUF_DIR, "bmo-qwen.Q4_K_M.gguf")
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


def step2_convert_gguf():
    """Convert merged model to GGUF format using llama.cpp."""
    print("\n" + "=" * 60)
    print("  Step 2: Converting to GGUF format")
    print("=" * 60)

    os.makedirs(GGUF_DIR, exist_ok=True)

    if os.path.exists(GGUF_FILE):
        print(f"  [SKIP] GGUF file already exists at {GGUF_FILE}")
        return

    # Try to find llama.cpp convert script
    convert_script = None
    search_paths = [
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        "/usr/local/bin/convert_hf_to_gguf.py",
        shutil.which("convert_hf_to_gguf.py"),
    ]

    for path in search_paths:
        if path and os.path.exists(path):
            convert_script = path
            break

    if not convert_script:
        print("  [INFO] llama.cpp convert script not found. Installing...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            pass

        # Clone llama.cpp if needed
        llama_cpp_dir = os.path.expanduser("~/llama.cpp")
        if not os.path.exists(llama_cpp_dir):
            print("  Cloning llama.cpp for GGUF conversion...")
            subprocess.run(
                ["git", "clone", "--depth=1",
                 "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r",
                 os.path.join(llama_cpp_dir, "requirements.txt")],
                check=True, capture_output=True,
            )

        convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")

    if not os.path.exists(convert_script):
        print(f"  [ERROR] Cannot find convert script at {convert_script}")
        print("  Please install llama.cpp manually:")
        print("    git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp")
        print("    pip install -r ~/llama.cpp/requirements.txt")
        sys.exit(1)

    # Convert to FP16 GGUF first
    fp16_gguf = os.path.join(GGUF_DIR, "bmo-qwen.f16.gguf")
    print(f"  Converting to GGUF (FP16)...")
    subprocess.run(
        [sys.executable, convert_script, MERGED_DIR,
         "--outfile", fp16_gguf, "--outtype", "f16"],
        check=True,
    )

    # Quantize to Q4_K_M for Ollama (good quality/size tradeoff)
    quantize_bin = None
    quant_search = [
        os.path.expanduser("~/llama.cpp/build/bin/llama-quantize"),
        os.path.expanduser("~/llama.cpp/llama-quantize"),
        shutil.which("llama-quantize"),
    ]
    for path in quant_search:
        if path and os.path.exists(path):
            quantize_bin = path
            break

    if quantize_bin:
        print(f"  Quantizing to Q4_K_M...")
        subprocess.run(
            [quantize_bin, fp16_gguf, GGUF_FILE, "Q4_K_M"],
            check=True,
        )
        # Remove intermediate FP16 file
        os.remove(fp16_gguf)
        print(f"  [OK] GGUF saved: {GGUF_FILE}")
    else:
        # If no quantize binary, use FP16 directly
        print("  [WARNING] llama-quantize not found, using FP16 GGUF directly")
        print("  To build: cd ~/llama.cpp && cmake -B build && cmake --build build --config Release")
        os.rename(fp16_gguf, GGUF_FILE.replace("Q4_K_M", "f16"))
        print(f"  [OK] GGUF saved (FP16): {GGUF_FILE.replace('Q4_K_M', 'f16')}")


def step3_create_ollama_model():
    """Create Ollama Modelfile and import the model."""
    print("\n" + "=" * 60)
    print("  Step 3: Creating Ollama model")
    print("=" * 60)

    # Find the actual GGUF file
    gguf = GGUF_FILE
    if not os.path.exists(gguf):
        # Check for f16 variant
        f16_gguf = GGUF_FILE.replace("Q4_K_M", "f16")
        if os.path.exists(f16_gguf):
            gguf = f16_gguf
        else:
            print(f"  [ERROR] No GGUF file found in {GGUF_DIR}")
            sys.exit(1)

    # Write Modelfile
    modelfile_content = f"""FROM {gguf}

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"

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

    # Create Ollama model
    print(f"  Creating Ollama model '{OLLAMA_MODEL_NAME}'...")
    result = subprocess.run(
        ["ollama", "create", OLLAMA_MODEL_NAME, "-f", MODELFILE_PATH],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print(f"  [OK] Ollama model '{OLLAMA_MODEL_NAME}' created successfully!")
        print(f"\n  To use with BMO, update config.json:")
        print(f'    "text_model": "{OLLAMA_MODEL_NAME}"')
        print(f"\n  Or test it directly:")
        print(f'    ollama run {OLLAMA_MODEL_NAME} "Who is BMO?"')
    else:
        print(f"  [ERROR] Failed to create Ollama model:")
        print(f"  {result.stderr}")


def main():
    if not os.path.exists(ADAPTER_DIR):
        print(f"[ERROR] Adapter not found at {ADAPTER_DIR}")
        print("  Run 3_train_qlora.py first!")
        sys.exit(1)

    step1_merge_adapter()
    step2_convert_gguf()
    step3_create_ollama_model()

    print("\n" + "=" * 60)
    print("  All done! Your fine-tuned BMO model is ready.")
    print("=" * 60)
    print(f"\n  Model name: {OLLAMA_MODEL_NAME}")
    print(f"  Update config.json: \"text_model\": \"{OLLAMA_MODEL_NAME}\"")


if __name__ == "__main__":
    main()
