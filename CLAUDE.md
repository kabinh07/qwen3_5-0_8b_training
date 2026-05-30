# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fine-tuning pipeline for `Qwen3.5-0.8B` (vision/OCR) using Unsloth + LoRA. Single training script (`train.py`) reads all config from environment variables. Containerised via Docker.

## Running training

```bash
# build image and train (default mode)
docker compose up --build

# other modes
docker compose run --rm trainer --mode eval
docker compose run --rm trainer --mode infer   # also set IMAGE_PATH in .env
docker compose run --rm trainer --mode export
```

No test suite. No linter config. No package manager — deps installed in Dockerfile.

## Environment variables

All config lives in `.env` (gitignored). `HF_TOKEN` is the only required var with no default. All others have defaults coded in `train.py` lines 44–78. Key vars:

| Var | Default | Notes |
|-----|---------|-------|
| `HF_TOKEN` | — | **Required** |
| `BASE_MODEL` | `unsloth/Qwen3.5-0.8B` | HF model ID or local path |
| `LOAD_IN_4BIT` | `false` | `true` halves VRAM, slower |
| `MAX_STEPS` | `500` | `-1` to use `NUM_TRAIN_EPOCHS` instead |
| `PER_DEVICE_BATCH_SIZE` | `16` | Lower if OOM |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Effective batch = batch × accum |
| `SAVE_MERGED_16BIT` / `SAVE_GGUF_Q8` / `SAVE_GGUF_Q4_K_M` | `true` | Export steps after training |

## Architecture

`train.py` has four modes dispatched from `main()`:

- **train** — downloads HF dataset → pairs `.txt`/`.jpg` files from `DATA_DIR` → builds chat-format dicts → trains with `SFTTrainer` + `UnslothVisionDataCollator` → saves LoRA → runs `mode_export`
- **eval** — loads val split, runs inference on up to 200 samples, prints GT vs prediction
- **infer** — single image via `IMAGE_PATH` env var, loads from `INFER_MODEL_PATH`
- **export** — saves merged 16-bit and/or GGUF Q8/Q4 from saved LoRA

Data flow: HuggingFace dataset (tar archives) → extracted to `DATA_DIR` → `*.txt` + `*.jpg` pairs → chat-format list in memory → `SFTTrainer`.

Outputs land in `./outputs/` (bind-mounted): `qwen_lora/` (LoRA adapter), `merged_16bit/`, `gguf_q8/`, `gguf_q4_k_m/`.

HF model/dataset cache is a named Docker volume (`huggingface_cache`) so it survives container rebuilds.

## Dockerfile

Extends `unsloth/unsloth:latest`. Adds `datasets` pip package. Copies `train.py` only — no source directory mounting. Rebuild image if `train.py` changes, or use `docker compose run` with a bind-mount override for faster iteration.
