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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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
MAX_STEPS                = env_int("MAX_STEPS", -1)         # -1 → use epochs
NUM_EPOCHS               = env_int("NUM_TRAIN_EPOCHS", 3)
EARLY_STOPPING_PATIENCE  = env_int("EARLY_STOPPING_PATIENCE", 3)
LR                = env_float("LEARNING_RATE", 2e-4)
WEIGHT_DECAY      = env_float("WEIGHT_DECAY", 0.001)
LR_SCHEDULER      = env("LR_SCHEDULER", "linear")
SEED              = env_int("SEED", 3407)
LOGGING_STEPS     = env_int("LOGGING_STEPS", 10)
EVAL_STEPS        = env_int("EVAL_STEPS", 100)
DATASET_NUM_PROC  = env_int("DATASET_NUM_PROC", 4)

SAVE_MERGED       = env_bool("SAVE_MERGED_16BIT", "true")
SAVE_GGUF_Q8      = env_bool("SAVE_GGUF_Q8", "true")
SAVE_GGUF_Q4      = env_bool("SAVE_GGUF_Q4_K_M", "true")

PUSH_TO_HUB       = env_bool("PUSH_TO_HUB", "false")
HF_REPO_ID        = env("HF_REPO_ID", "kavinh07/unsloth_finetune_qwen3.5-0.8B")

OCR_PROMPT        = env("OCR_PROMPT", "All text in this image is in {LANG}. Transcribe every character exactly as it appears. Output only the text.")

os.environ["HF_TOKEN"] = HF_TOKEN


# ─────────────────────────── dataset ─────────────────────────────────────────

class OCRDatasetPreparator:
    """
    Loads a parquet HF dataset (image / text / class_name / source columns),
    detects the language of each ground-truth label, injects it into the prompt
    template via the {LANG} placeholder, and returns chat-format records ready
    for UnslothVisionDataCollator.

    Language detection rules (Unicode):
      Bangla only          → "Bangla"          (U+0980–U+09FF)
      ASCII alpha only     → "English"
      Both present         → "Both Bangla and English"
    """

    _BANGLA_LO = "ঀ"
    _BANGLA_HI = "৿"

    def __init__(
        self,
        dataset_id: str,
        token: str,
        prompt_template: str,
        train_samples: int = 0,
        val_samples: int = 0,
    ) -> None:
        self.dataset_id = dataset_id
        self.token = token
        self.prompt_template = prompt_template
        self.train_samples = train_samples
        self.val_samples = val_samples

    def detect_lang(self, text: str) -> str:
        has_bangla  = any(self._BANGLA_LO <= ch <= self._BANGLA_HI for ch in text)
        has_english = any(ch.isascii() and ch.isalpha() for ch in text)
        if has_bangla and has_english:
            return "Both Bangla and English"
        if has_bangla:
            return "Bangla"
        return "English"

    def format_prompt(self, text: str) -> str:
        return self.prompt_template.replace("{LANG}", self.detect_lang(text))

    def _to_chat(self, record: Dict) -> Dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",  "text":  self.format_prompt(record["text"])},
                        {"type": "image", "image": record["image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": record["text"]}],
                },
            ],
        }

    def _select(self, split, n: int):
        return split.select(range(min(n, len(split)))) if n > 0 else split

    def load(self) -> Tuple[List[Dict], List[Dict]]:
        from datasets import load_dataset as hf_load

        print(f"[data] Loading {self.dataset_id} …")
        ds = hf_load(self.dataset_id, token=self.token)

        train_split = self._select(ds["train"],      self.train_samples)
        val_split   = self._select(ds["validation"], self.val_samples)

        print(f"[data] Converting {len(train_split)} train + {len(val_split)} val records …")
        train_data = [self._to_chat(r) for r in train_split]
        val_data   = [self._to_chat(r) for r in val_split]
        return train_data, val_data


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


def run_inference(model, tokenizer, image, instruction: str = None):
    """Run inference on a single image (PIL Image or path string)."""
    from unsloth import FastVisionModel
    from transformers import TextStreamer

    if instruction is None:
        instruction = OCR_PROMPT.replace("{LANG}", "Both Bangla and English")

    FastVisionModel.for_inference(model)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
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

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        import jiwer
        pred_ids, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        pred_strs = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_strs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        cer = jiwer.cer(label_strs, pred_strs)
        return {"cer": cer}
    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def mode_train():
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback

    # ── data ──
    training_data, val_data = OCRDatasetPreparator(
        dataset_id=HF_DATASET,
        token=HF_TOKEN,
        prompt_template=OCR_PROMPT,
        train_samples=TRAIN_SAMPLES,
        val_samples=VAL_SAMPLES,
    ).load()
    print(f"[train] Training samples  : {len(training_data)}")
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
        eval_strategy="steps" if len(val_data) > 0 else "no",
        eval_steps=EVAL_STEPS,
        save_strategy="steps" if len(val_data) > 0 else "no",
        save_steps=EVAL_STEPS,
        load_best_model_at_end=len(val_data) > 0,
        metric_for_best_model="cer",
        greater_is_better=False,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        seed=SEED,
        output_dir=str(OUTPUT_DIR),
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
        dataset_num_proc=DATASET_NUM_PROC,
    )
    if MAX_STEPS > 0:
        sft_args["max_steps"] = MAX_STEPS
    else:
        sft_args["num_train_epochs"] = NUM_EPOCHS

    has_val = len(val_data) > 0
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=training_data,
        eval_dataset=val_data if has_val else None,
        compute_metrics=make_compute_metrics(tokenizer) if has_val else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if has_val else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)] if has_val else [],
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

    # ── push best model to Hub ──
    if PUSH_TO_HUB:
        print(f"[train] Pushing LoRA adapter to Hub → {HF_REPO_ID}")
        model.push_to_hub_merged(HF_REPO_ID, tokenizer, save_method="lora", token=HF_TOKEN)
        print(f"[train] Hub push complete → https://huggingface.co/{HF_REPO_ID}")

    # ── quick sanity-check inference ──
    print("[train] Running sanity-check inference …")
    sample = training_data[0]["messages"][0]["content"]
    result = run_inference(model, tokenizer, sample[1]["image"], instruction=sample[0]["text"])
    print(f"[infer] {result}")

    mode_export(model, tokenizer)


def mode_eval():
    from unsloth import FastVisionModel

    _, eval_data = OCRDatasetPreparator(
        dataset_id=HF_DATASET,
        token=HF_TOKEN,
        prompt_template=OCR_PROMPT,
        val_samples=VAL_SAMPLES,
    ).load()

    model, tokenizer = load_model_and_tokenizer(for_inference=True)
    FastVisionModel.for_inference(model)

    print(f"[eval] Evaluating on {min(200, len(eval_data))} samples …")
    for item in eval_data[:200]:
        content      = item["messages"][0]["content"]
        ground_truth = item["messages"][1]["content"][0]["text"]
        prediction   = run_inference(model, tokenizer, content[1]["image"], instruction=content[0]["text"])
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
        try:
            model.save_pretrained_gguf(gguf_q8_path, tokenizer)
        except Exception as e:
            print(f"[export] GGUF Q8_0 skipped — vision models may not support GGUF: {e}")

    if SAVE_GGUF_Q4:
        gguf_q4_path = str(OUTPUT_DIR / "gguf_q4_k_m")
        print(f"[export] Saving GGUF q4_k_m → {gguf_q4_path}")
        try:
            model.save_pretrained_gguf(gguf_q4_path, tokenizer, quantization_method="q4_k_m")
        except Exception as e:
            print(f"[export] GGUF q4_k_m skipped — vision models may not support GGUF: {e}")

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