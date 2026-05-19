#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.5-0.8B Vision OCR Fine-tuning
Portable training script — reads all config from environment variables.
Usage:
    python train.py                   # train
    python train.py --mode eval       # run eval loop only
    python train.py --mode infer      # single-image inference test
    python train.py --mode export     # export saved LoRA to GGUF / merged
"""

import argparse
import os
import sys
import tarfile
from pathlib import Path
from typing import Dict, List


# ─────────────────────────── helpers ─────────────────────────────────────────

def env(key: str, default=None):
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"Required env var '{key}' is not set.")
    return val


def env_bool(key: str, default: str = "false") -> bool:
    return os.environ.get(key, default).strip().lower() in ("1", "true", "yes")


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


# ─────────────────────────── config ──────────────────────────────────────────

HF_TOKEN          = env("HF_TOKEN")
HF_DATASET        = env("HF_DATASET", "kavinh07/synth-200k-ocr")
DATA_DIR          = Path(env("DATA_DIR", "/workspace/data"))
OUTPUT_DIR        = Path(env("OUTPUT_DIR", "/workspace/outputs"))
MODEL_SAVE_DIR    = Path(env("MODEL_SAVE_DIR", "/workspace/outputs/qwen_lora"))

BASE_MODEL        = env("BASE_MODEL", "unsloth/Qwen3.5-0.8B")
LOAD_IN_4BIT      = env_bool("LOAD_IN_4BIT", "false")

LORA_R            = env_int("LORA_R", 16)
LORA_ALPHA        = env_int("LORA_ALPHA", 16)
LORA_DROPOUT      = env_float("LORA_DROPOUT", 0.0)

TRAIN_SAMPLES     = env_int("TRAIN_SAMPLES", 80000)
VAL_SAMPLES       = env_int("VAL_SAMPLES", 5000)
BATCH_SIZE        = env_int("PER_DEVICE_BATCH_SIZE", 16)
EVAL_BATCH_SIZE   = env_int("PER_DEVICE_EVAL_BATCH_SIZE", BATCH_SIZE)
GRAD_ACCUM        = env_int("GRADIENT_ACCUMULATION_STEPS", 4)
WARMUP_STEPS      = env_int("WARMUP_STEPS", 5)
MAX_STEPS         = env_int("MAX_STEPS", 500)      # -1 → use epochs
NUM_EPOCHS        = env_int("NUM_TRAIN_EPOCHS", 1)
LR                = env_float("LEARNING_RATE", 2e-4)
WEIGHT_DECAY      = env_float("WEIGHT_DECAY", 0.001)
LR_SCHEDULER      = env("LR_SCHEDULER", "linear")
MAX_LENGTH        = env_int("MAX_LENGTH", 1024)
SEED              = env_int("SEED", 3407)
LOGGING_STEPS     = env_int("LOGGING_STEPS", 10)
EVAL_STEPS        = env_int("EVAL_STEPS", 100)
DATASET_NUM_PROC  = env_int("DATASET_NUM_PROC", 4)

SAVE_MERGED       = env_bool("SAVE_MERGED_16BIT", "true")
SAVE_GGUF_Q8      = env_bool("SAVE_GGUF_Q8", "true")
SAVE_GGUF_Q4      = env_bool("SAVE_GGUF_Q4_K_M", "true")

OCR_PROMPT        = env("OCR_PROMPT", "Extract all text from this image. Output only the text.")

os.environ["HF_TOKEN"] = HF_TOKEN


# ─────────────────────────── data helpers ────────────────────────────────────

def download_dataset():
    """Download and extract the HuggingFace dataset."""
    from huggingface_hub import snapshot_download

    raw_dir = DATA_DIR.parent / "raw_dataset"
    print(f"[data] Downloading {HF_DATASET} → {raw_dir}")
    snapshot_download(HF_DATASET, repo_type="dataset", local_dir=str(raw_dir),
                      token=HF_TOKEN)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tar_files = list(raw_dir.glob("*.tar"))
    if not tar_files:
        print("[data] No .tar files found — assuming dataset is already extracted.")
        return

    print(f"[data] Extracting {len(tar_files)} archive(s) → {DATA_DIR}")
    for tar_path in tar_files:
        with tarfile.open(tar_path) as tf:
            tf.extractall(path=str(DATA_DIR))
    print("[data] Extraction complete.")


def load_records() -> List[Dict]:
    """Read every .txt file and pair it with its .jpg counterpart."""
    records = []
    for txt_path in DATA_DIR.glob("*.txt"):
        img_path = DATA_DIR / (txt_path.stem + ".jpg")
        if not img_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        records.append({"image_path": str(img_path), "text": text})
    print(f"[data] Loaded {len(records)} image-text pairs.")
    return records


def to_chat_format(records: List[Dict], prompt: str = OCR_PROMPT) -> List[Dict]:
    """Convert raw records to the Qwen3.5 multi-modal chat format."""
    chat = []
    for item in records:
        chat.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": item["image_path"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["text"]}],
                },
            ],
            "image_path": item["image_path"],
        })
    return chat


# ─────────────────────────── model helpers ───────────────────────────────────

def load_model_and_tokenizer(for_inference: bool = False):
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL,
        load_in_4bit=LOAD_IN_4BIT,
        use_gradient_checkpointing="unsloth",
    )

    if not for_inference:
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=env_bool("FINETUNE_VISION_LAYERS", "true"),
            finetune_language_layers=env_bool("FINETUNE_LANGUAGE_LAYERS", "true"),
            finetune_attention_modules=env_bool("FINETUNE_ATTENTION_MODULES", "true"),
            finetune_mlp_modules=env_bool("FINETUNE_MLP_MODULES", "true"),
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            random_state=SEED,
            use_rslora=False,
            loftq_config=None,
            target_modules="all-linear",
        )

    return model, tokenizer


def run_inference(model, tokenizer, image_path: str, instruction: str = OCR_PROMPT):
    """Run inference on a single image and return the decoded text."""
    from unsloth import FastVisionModel
    from transformers import TextStreamer

    FastVisionModel.for_inference(model)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image_path,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    out_ids = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=512,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )
    out_ids = out_ids[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(out_ids, skip_special_tokens=True)


# ─────────────────────────── modes ───────────────────────────────────────────

def mode_train():
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    # ── data ──
    download_dataset()
    records = load_records()
    train_records = records[:TRAIN_SAMPLES]
    val_records = records[TRAIN_SAMPLES: TRAIN_SAMPLES + VAL_SAMPLES]
    training_data = to_chat_format(train_records)
    val_data = to_chat_format(val_records)
    print(f"[train] Training samples: {len(training_data)}")
    print(f"[train] Validation samples: {len(val_data)}")

    # ── model ──
    model, tokenizer = load_model_and_tokenizer(for_inference=False)
    FastVisionModel.for_training(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── trainer ──
    sft_args = dict(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps" if len(val_data) > 0 else "no",
        eval_steps=EVAL_STEPS,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        seed=SEED,
        output_dir=str(OUTPUT_DIR),
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_LENGTH,
        dataset_num_proc=DATASET_NUM_PROC,
    )
    if MAX_STEPS > 0:
        sft_args["max_steps"] = MAX_STEPS
    else:
        sft_args["num_train_epochs"] = NUM_EPOCHS

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=training_data,
        eval_dataset=val_data if len(val_data) > 0 else None,
        args=SFTConfig(**sft_args),
    )

    print("[train] Starting training …")
    trainer_stats = trainer.train()
    print(f"[train] Done. Stats: {trainer_stats}")

    # ── save LoRA ──
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_SAVE_DIR))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
    print(f"[train] LoRA saved → {MODEL_SAVE_DIR}")

    # ── quick sanity-check inference ──
    print("[train] Running sanity-check inference …")
    result = run_inference(model, tokenizer, training_data[0]["image_path"])
    print(f"[infer] {result}")

    mode_export(model, tokenizer)


def mode_eval():
    from unsloth import FastVisionModel

    download_dataset()
    records = load_records()
    eval_records = records[TRAIN_SAMPLES: TRAIN_SAMPLES + VAL_SAMPLES]
    if not eval_records:
        eval_records = records[TRAIN_SAMPLES:]
    eval_data = to_chat_format(eval_records)

    model, tokenizer = load_model_and_tokenizer(for_inference=True)
    FastVisionModel.for_inference(model)

    print(f"[eval] Evaluating on {min(200, len(eval_data))} samples …")
    for item in eval_data[:200]:
        ground_truth = item["messages"][1]["content"][0]["text"]
        prediction = run_inference(model, tokenizer, item["image_path"])
        print(f"GT : {ground_truth}")
        print(f"PRD: {prediction}")
        print("─" * 60)


def mode_infer():
    """Single-image inference — pass IMAGE_PATH env var."""
    image_path = os.environ.get("IMAGE_PATH")
    if not image_path:
        print("Set IMAGE_PATH env var to the image you want to run inference on.")
        sys.exit(1)

    model_path = os.environ.get("INFER_MODEL_PATH", str(MODEL_SAVE_DIR))
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path,
        load_in_4bit=LOAD_IN_4BIT,
    )
    result = run_inference(model, tokenizer, image_path)
    print(result)


def mode_export(model=None, tokenizer=None):
    """Export LoRA to merged 16-bit and/or GGUF quantisations."""
    if model is None:
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=str(MODEL_SAVE_DIR),
            load_in_4bit=LOAD_IN_4BIT,
        )
        from unsloth import FastVisionModel as _FVM
        _FVM.for_inference(model)

    if SAVE_MERGED:
        merged_path = str(OUTPUT_DIR / "merged_16bit")
        print(f"[export] Saving merged 16-bit model → {merged_path}")
        model.save_pretrained_merged(merged_path, tokenizer)

    if SAVE_GGUF_Q8:
        gguf_q8_path = str(OUTPUT_DIR / "gguf_q8")
        print(f"[export] Saving GGUF Q8_0 → {gguf_q8_path}")
        model.save_pretrained_gguf(gguf_q8_path, tokenizer)

    if SAVE_GGUF_Q4:
        gguf_q4_path = str(OUTPUT_DIR / "gguf_q4_k_m")
        print(f"[export] Saving GGUF q4_k_m → {gguf_q4_path}")
        model.save_pretrained_gguf(gguf_q4_path, tokenizer, quantization_method="q4_k_m")

    print("[export] Export complete.")


# ─────────────────────────── entry point ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B Vision fine-tuner")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "infer", "export"],
        default="train",
        help="Execution mode (default: train)",
    )
    args = parser.parse_args()

    print(f"[main] Mode: {args.mode}")
    print(f"[main] Base model : {BASE_MODEL}")
    print(f"[main] 4-bit LoRA : {LOAD_IN_4BIT}")
    print(f"[main] Data dir   : {DATA_DIR}")
    print(f"[main] Output dir : {OUTPUT_DIR}")

    if args.mode == "train":
        mode_train()
    elif args.mode == "eval":
        mode_eval()
    elif args.mode == "infer":
        mode_infer()
    elif args.mode == "export":
        mode_export()


if __name__ == "__main__":
    main()