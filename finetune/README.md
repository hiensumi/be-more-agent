# Adventure Time Wiki → QLoRA Fine-Tuning Pipeline

Fine-tune `qwen2.5:3b` on Adventure Time wiki knowledge so BMO can answer
questions about the Land of Ooo without web searches.

## Hardware Requirements

- **GPU**: NVIDIA GPU with 4+ GB VRAM (tested on T1200 4GB)
- **RAM**: 16+ GB system RAM
- **Disk**: ~10 GB free (model downloads + training artifacts)

## Quick Start

```bash
# 1. Install dependencies
pip install -r finetune/requirements.txt

# 2. Scrape the Adventure Time wiki (~15-30 min)
python finetune/1_scrape_wiki.py

# 3. Clean & format training data (~1 min)
python finetune/2_format_data.py

# 4. Fine-tune with QLoRA (~2-6 hours on 4GB GPU)
python finetune/3_train_qlora.py

# 5. Export to Ollama (~10-20 min)
python finetune/4_export_ollama.py

# 6. Update BMO's config to use the new model
# Edit config.json: "text_model": "bmo-qwen"
```

## Pipeline Overview

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `1_scrape_wiki.py` | Wiki API | `raw_wiki_pages.json` |
| 2 | `2_format_data.py` | `raw_wiki_pages.json` | `train_data.jsonl` |
| 3 | `3_train_qlora.py` | `train_data.jsonl` | `bmo-qwen-qlora/` |
| 4 | `4_export_ollama.py` | `bmo-qwen-qlora/` | Ollama model `bmo-qwen` |

## Configuration

Training hyperparameters can be adjusted in `3_train_qlora.py`:

- `QLORA_R`: LoRA rank (default 16, lower = less memory)
- `MAX_SEQ_LENGTH`: Max token length (default 512, lower = less VRAM)
- `BATCH_SIZE`: Per-device batch (default 1 for 4GB VRAM)
- `GRADIENT_ACCUMULATION`: Steps to accumulate (default 8 → effective batch 8)
- `NUM_EPOCHS`: Training epochs (default 3)
- `LEARNING_RATE`: LR for AdamW (default 2e-4)

### If you run out of VRAM:

1. Reduce `MAX_SEQ_LENGTH` to 256
2. Reduce `QLORA_R` to 8
3. Set `packing=False` in the SFTTrainer call

### If you have more VRAM (8+ GB):

1. Increase `MAX_SEQ_LENGTH` to 1024
2. Increase `QLORA_R` to 32
3. Increase `BATCH_SIZE` to 2

## Using Google Colab (Recommended for faster training)

If your local GPU is too slow, upload the training data and run step 3 on Colab:

1. Run steps 1-2 locally
2. Upload `train_data.jsonl` to Colab
3. Run step 3 on Colab (free T4 GPU)
4. Download `bmo-qwen-qlora/` folder
5. Run step 4 locally to export to Ollama
