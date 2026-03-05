#!/usr/bin/env python3
"""Step 3: Fine-tune Qwen2.5:3B with QLoRA on Adventure Time wiki data.

Uses 4-bit quantization + LoRA adapters for memory-efficient training.
Optimized for 4GB VRAM (NVIDIA T1200) with aggressive memory settings.

Requires: torch, transformers, peft, bitsandbytes, datasets, trl, accelerate

Input:  finetune/train_data.jsonl
Output: finetune/bmo-qwen-qlora/ (adapter weights)
"""

import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "train_data.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "bmo-qwen-qlora")

# ── Model ────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ── QLoRA config ─────────────────────────────────────────────────────────────

QLORA_R = 16            # LoRA rank — lower = less memory, 16 is a good balance
QLORA_ALPHA = 32        # LoRA alpha — typically 2x rank
QLORA_DROPOUT = 0.05    # Small dropout for regularization
QLORA_TARGET_MODULES = [  # Which layers to adapt
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training hyperparameters (tuned for 4GB VRAM) ────────────────────────────

MAX_SEQ_LENGTH = 512          # Keep short to fit in VRAM
BATCH_SIZE = 1                # Minimum batch for 4GB
GRADIENT_ACCUMULATION = 8     # Effective batch = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STEPS = 100
MAX_GRAD_NORM = 0.3


def load_data():
    """Load JSONL training data."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Training data not found at {DATA_FILE}\n"
            "Run 1_scrape_wiki.py then 2_format_data.py first!"
        )

    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[DATA] Loaded {len(records)} training examples")
    return records


def format_conversation(example):
    """Format conversations into Qwen ChatML template string."""
    convs = example["conversations"]
    text_parts = []

    for msg in convs:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            text_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            text_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    return {"text": "\n".join(text_parts)}


def main():
    print("=" * 60)
    print("  BMO QLoRA Fine-Tuning — Adventure Time Knowledge")
    print("=" * 60)

    # ── Check GPU ────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"\n[GPU] {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\n[WARNING] No GPU detected! Training will be extremely slow on CPU.")
        print("  Consider using Google Colab or a cloud GPU instance.")

    # ── Load data ────────────────────────────────────────────────
    records = load_data()
    dataset = Dataset.from_list(records)
    dataset = dataset.map(format_conversation, remove_columns=["conversations"])

    # Train/eval split (95/5)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"[DATA] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ── Quantization config ──────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings
    )

    # ── Load model + tokenizer ───────────────────────────────────
    print(f"\n[MODEL] Loading {MODEL_NAME} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── LoRA config ──────────────────────────────────────────────
    lora_config = LoraConfig(
        r=QLORA_R,
        lora_alpha=QLORA_ALPHA,
        lora_dropout=QLORA_DROPOUT,
        target_modules=QLORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[LORA] Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)")

    # ── Training args ────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_pin_memory=False,  # Save memory
        remove_unused_columns=False,
    )

    # ── Trainer ──────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,  # Pack multiple short examples into one sequence
    )

    # ── Train! ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Starting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"{'=' * 60}\n")

    trainer.train()

    # ── Save ─────────────────────────────────────────────────────
    print(f"\n[SAVE] Saving adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[DONE] Fine-tuning complete!")
    print(f"\n  Next step: run 4_export_ollama.py to merge & export to Ollama")


if __name__ == "__main__":
    main()
