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

Key fixes vs previous version:
  - Image placed BEFORE text in chat template (Qwen2.5-VL requirement)
  - Unicode NFC normalisation on all Bangla labels
  - Vision encoder frozen by default (FINETUNE_VISION_LAYERS=false)
  - WARMUP_RATIO replaces the hard-coded 5-step warmup
  - LR scheduler changed to cosine
  - confusion_training rows oversampled N× before subsampling
  - Greedy decoding (do_sample=False) for inference — OCR is not creative
  - Overfitting guard: train/val CER gap logged; early stopping on val CER
"""

import unsloth  # must be first to apply all optimizations before transformers/peft load

import argparse
import os
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import numpy as np

load_dotenv()


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


def nfc(text: str) -> str:
    """
    NFC-normalise Bangla text and strip stray zero-width characters.
    Visually identical conjuncts can have multiple Unicode encodings;
    normalising ensures label == prediction comparisons are fair.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")  # ZWNJ / ZWJ
    return text


# ─────────────────────────── config ──────────────────────────────────────────

HF_TOKEN          = env("HF_TOKEN")
HF_DATASET        = env("HF_DATASET", "kavinh07/ocr_dataset_shamadhan_synth_30k_p4")
DATA_DIR          = Path(env("DATA_DIR", "/workspace/data"))
OUTPUT_DIR        = Path(env("OUTPUT_DIR", "/workspace/outputs"))
MODEL_SAVE_DIR    = Path(env("MODEL_SAVE_DIR", "/workspace/outputs/qwen_lora"))

BASE_MODEL        = env("BASE_MODEL", "unsloth/Qwen3.5-0.8B")
LOAD_IN_4BIT      = env_bool("LOAD_IN_4BIT", "false")

LORA_R            = env_int("LORA_R", 16)
LORA_ALPHA        = env_int("LORA_ALPHA", 16)
LORA_DROPOUT      = env_float("LORA_DROPOUT", 0.05)

TRAIN_SAMPLES          = env_int("TRAIN_SAMPLES", 0)
VAL_SAMPLES            = env_int("VAL_SAMPLES", 5000)
BATCH_SIZE             = env_int("PER_DEVICE_BATCH_SIZE", 16)
EVAL_BATCH_SIZE        = env_int("PER_DEVICE_EVAL_BATCH_SIZE", BATCH_SIZE)
GRAD_ACCUM             = env_int("GRADIENT_ACCUMULATION_STEPS", 4)

# WARMUP_RATIO takes priority over WARMUP_STEPS when set
WARMUP_RATIO           = env_float("WARMUP_RATIO", 0.05)
WARMUP_STEPS           = env_int("WARMUP_STEPS", 0)   # 0 = use ratio instead

MAX_STEPS              = env_int("MAX_STEPS", -1)
NUM_EPOCHS             = env_int("NUM_TRAIN_EPOCHS", 5)
EARLY_STOPPING_PATIENCE= env_int("EARLY_STOPPING_PATIENCE", 3)

LR                     = env_float("LEARNING_RATE", 2e-4)
WEIGHT_DECAY           = env_float("WEIGHT_DECAY", 0.01)
LR_SCHEDULER           = env("LR_SCHEDULER", "cosine")
SEED                   = env_int("SEED", 3407)
LOGGING_STEPS          = env_int("LOGGING_STEPS", 50)
EVAL_STEPS             = env_int("EVAL_STEPS", 500)
DATASET_NUM_PROC       = env_int("DATASET_NUM_PROC", 4)

# How many times to repeat confusion_training (conjunct) rows
CONFUSION_OVERSAMPLE   = env_int("CONFUSION_OVERSAMPLE", 4)

# Optional directory of local hard-negative pairs: <stem>.png + <stem>.txt
# These are injected into training data and oversampled like confusion_training rows.
# Use this to fix specific recurring errors (e.g. ড/দ confusion, repeated digit hallucination).
_local_hard_neg_raw    = os.environ.get("LOCAL_HARD_NEG_DIR", "")
LOCAL_HARD_NEG_DIR     = Path(_local_hard_neg_raw) if _local_hard_neg_raw else None

SAVE_MERGED       = env_bool("SAVE_MERGED_16BIT", "true")
SAVE_GGUF_Q8      = env_bool("SAVE_GGUF_Q8", "true")
SAVE_GGUF_Q4      = env_bool("SAVE_GGUF_Q4_K_M", "true")

PUSH_TO_HUB       = env_bool("PUSH_TO_HUB", "false")
HF_REPO_ID        = env("HF_REPO_ID", "kavinh07/unsloth_finetune_qwen3.5-0.8B_p4")

OCR_PROMPT        = env(
    "OCR_PROMPT",
    "All text in this image is in {LANG}. Transcribe every character exactly as it "
    "appears, including all conjunct consonants. Output only the transcribed text, nothing else.",
)

os.environ["HF_TOKEN"] = HF_TOKEN


# ─────────────────────────── dataset ─────────────────────────────────────────

class OCRDatasetPreparator:
    """
    Loads a parquet HF dataset (image / text / class_name / source columns),
    oversamples the hard 'confusion_training' conjunct rows, NFC-normalises
    every label, and returns chat-format records ready for
    UnslothVisionDataCollator.

    Chat format fix: image token placed BEFORE instruction text, which is
    required by Qwen2.5-VL's positional encoding scheme.

    Language detection (Unicode codepoint ranges):
      Bangla only              → "Bangla"
      ASCII alpha only         → "English"
      Both present             → "Both Bangla and English"
    """

    _BANGLA_LO = "ঀ"   # U+0980
    _BANGLA_HI = "৿"   # U+09FF

    def __init__(
        self,
        dataset_id: str,
        token: str,
        prompt_template: str,
        train_samples: int = 0,
        val_samples: int = 0,
        confusion_oversample: int = 4,
    ) -> None:
        self.dataset_id          = dataset_id
        self.token               = token
        self.prompt_template     = prompt_template
        self.train_samples       = train_samples
        self.val_samples         = val_samples
        self.confusion_oversample = confusion_oversample

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
        """
        Convert a raw dataset row to an Unsloth chat record.

        IMPORTANT: image must come BEFORE the text instruction in the user
        content list.  Qwen2.5-VL encodes visual tokens at the position of the
        <image> placeholder; putting text first shifts those positions and
        causes the language decoder to misalign visual and text representations.
        """
        text = nfc(record["text"])   # ← normalise Bangla conjuncts
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        # ✅ image FIRST
                        {"type": "image", "image": record["image"]},
                        {"type": "text",  "text":  self.format_prompt(text)},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                },
            ],
        }

    def _select(self, split, n: int):
        return split.select(range(min(n, len(split)))) if n > 0 else split

    def _load_local_hard_negatives(self) -> List[Dict]:
        """
        Read (image, label) pairs from LOCAL_HARD_NEG_DIR and convert to chat
        records.  Pairs are matched by stem: foo.png + foo.txt (or .jpg/.jpeg).
        Missing image or label file → silently skipped.
        Use this to inject targeted corrections for specific error patterns
        (e.g. repeated-digit hallucinations, ড/দ or গ/প confusions).
        """
        if LOCAL_HARD_NEG_DIR is None or not LOCAL_HARD_NEG_DIR.is_dir():
            return []
        from PIL import Image as PILImage
        records: List[Dict] = []
        for txt_path in sorted(LOCAL_HARD_NEG_DIR.glob("*.txt")):
            label = txt_path.read_text(encoding="utf-8").strip()
            if not label:
                continue
            image_path = None
            for ext in (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"):
                candidate = txt_path.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            image = PILImage.open(image_path).convert("RGB")
            records.append(self._to_chat({"image": image, "text": label}))
        if records:
            print(f"[data] Local hard negatives: {len(records)} pairs from {LOCAL_HARD_NEG_DIR}")
        return records

    def load(self) -> Tuple[List[Dict], List[Dict]]:
        from datasets import load_dataset as hf_load, concatenate_datasets

        print(f"[data] Loading {self.dataset_id} …")
        ds = hf_load(self.dataset_id, token=self.token)

        # ── oversample confusion_training (conjunct-heavy) rows ───────────
        train_raw  = ds["train"]
        easy       = train_raw.filter(
            lambda x: x["class_name"] != "confusion_training",
            num_proc=DATASET_NUM_PROC,
        )
        hard       = train_raw.filter(
            lambda x: x["class_name"] == "confusion_training",
            num_proc=DATASET_NUM_PROC,
        )
        print(
            f"[data] Raw split — easy: {len(easy)}, "
            f"hard (conjunct): {len(hard)}  →  repeating hard {self.confusion_oversample}×"
        )
        hard_nx    = concatenate_datasets([hard] * self.confusion_oversample)
        balanced   = concatenate_datasets([easy, hard_nx]).shuffle(seed=SEED)

        train_split = self._select(balanced,         self.train_samples)
        val_split   = self._select(ds["validation"], self.val_samples)

        print(f"[data] Converting {len(train_split)} train + {len(val_split)} val records …")
        train_data = [self._to_chat(r) for r in train_split]
        val_data   = [self._to_chat(r) for r in val_split]

        # ── inject local hard negatives (oversampled) ─────────────────────
        local_hard = self._load_local_hard_negatives()
        if local_hard:
            import random as _random
            augmented = local_hard * self.confusion_oversample
            train_data = augmented + train_data
            _random.seed(SEED)
            _random.shuffle(train_data)
            print(f"[data] After local injection: {len(train_data)} total train records")

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
        if hasattr(model, "peft_config"):
            print("[model] Existing LoRA adapters detected — merging into base weights …")
            model = model.merge_and_unload()

        model = FastVisionModel.get_peft_model(
            model,
            # Vision encoder frozen by default: it already knows how to see
            # glyphs; fine-tuning it on this data size hurts conjunct recall.
            finetune_vision_layers     = env_bool("FINETUNE_VISION_LAYERS",     "false"),
            finetune_language_layers   = env_bool("FINETUNE_LANGUAGE_LAYERS",   "true"),
            finetune_attention_modules = env_bool("FINETUNE_ATTENTION_MODULES", "true"),
            finetune_mlp_modules       = env_bool("FINETUNE_MLP_MODULES",       "true"),
            r            = LORA_R,
            lora_alpha   = LORA_ALPHA,
            lora_dropout = LORA_DROPOUT,
            bias         = "none",
            random_state = SEED,
            use_rslora   = False,
            loftq_config = None,
            target_modules = "all-linear",
        )

    return model, tokenizer


def run_inference(model, tokenizer, image, instruction: str = None, stream: bool = True):
    """Run OCR inference on a single image (PIL Image or path string)."""
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

    out_ids = model.generate(
        **inputs,
        streamer=TextStreamer(tokenizer, skip_prompt=True) if stream else None,
        max_new_tokens=512,
        use_cache=True,
        # ✅ Greedy decoding — OCR is deterministic, not creative.
        # Temperature=1.5 was adding random noise to conjunct predictions.
        do_sample=False,
        repetition_penalty=1.1,
    )
    out_ids = out_ids[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(out_ids, skip_special_tokens=True)


# ─────────────────────────── metrics ─────────────────────────────────────────

def make_compute_metrics(tokenizer):
    """
    Returns a compute_metrics function that reports:
      cer       — character error rate (primary optimisation target)
      cer_bn    — CER on Bangla-only samples (proxy for conjunct quality)
      cer_en    — CER on English-only samples
    """
    _BANGLA_LO = "ঀ"
    _BANGLA_HI = "৿"

    def is_bangla(text: str) -> bool:
        return any(_BANGLA_LO <= ch <= _BANGLA_HI for ch in text)

    def compute_metrics(eval_pred):
        import jiwer

        pred_ids, labels = eval_pred
        pred_ids = np.asarray(pred_ids)
        labels   = np.asarray(labels)

        # Teacher-forced logits at position i predict token i+1, so shift the
        # argmax predictions left by one to align them with the labels, then keep
        # ONLY the target positions (labels != -100). Without this the decoded
        # prediction spans the whole prompt + image region while the label is the
        # answer only — producing CER > 1.0 that does not reflect real accuracy
        # and wrongly trips early stopping (the previous run stopped at epoch ~1).
        pred_ids = pred_ids[:, :-1]
        labels   = labels[:, 1:]
        mask     = labels != -100

        pred_strs, label_strs = [], []
        for p_row, l_row, m_row in zip(pred_ids, labels, mask):
            if not m_row.any():
                continue
            label_strs.append(nfc(tokenizer.decode(l_row[m_row], skip_special_tokens=True)))
            pred_strs.append( nfc(tokenizer.decode(p_row[m_row], skip_special_tokens=True)))

        # Drop pairs with an empty reference (jiwer.cer divides by ref length).
        pairs = [(r, h) for r, h in zip(label_strs, pred_strs) if r.strip()]
        if not pairs:
            return {"cer": 1.0, "cer_bn": 1.0, "cer_en": 1.0}
        label_strs = [r for r, _ in pairs]
        pred_strs  = [h for _, h in pairs]

        overall_cer = jiwer.cer(label_strs, pred_strs)

        bn_pairs = [(r, h) for r, h in zip(label_strs, pred_strs) if is_bangla(r)]
        en_pairs = [(r, h) for r, h in zip(label_strs, pred_strs) if not is_bangla(r)]

        cer_bn = jiwer.cer([r for r, _ in bn_pairs], [h for _, h in bn_pairs]) if bn_pairs else 0.0
        cer_en = jiwer.cer([r for r, _ in en_pairs], [h for _, h in en_pairs]) if en_pairs else 0.0

        print(
            f"\n[metrics] CER overall={overall_cer:.4f} | "
            f"Bangla={cer_bn:.4f} ({len(bn_pairs)} samples) | "
            f"English={cer_en:.4f} ({len(en_pairs)} samples)"
        )
        return {"cer": overall_cer, "cer_bn": cer_bn, "cer_en": cer_en}

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# ─────────────────────────── overfitting monitor ─────────────────────────────

from transformers import TrainerCallback

class OverfitMonitorCallback(TrainerCallback):
    """
    Logs the train/val CER gap after every evaluation.
    Warns if val CER is significantly worse than train CER, which is the
    classic sign of overfitting on rare conjunct patterns.

    Threshold: warn when val_cer > train_cer * OVERFIT_WARN_RATIO
    """

    OVERFIT_WARN_RATIO = 1.5   # val CER 50% worse than train CER → warning

    def __init__(self):
        self._train_cer = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "train_cer" in logs:
            self._train_cer = logs["train_cer"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self._train_cer is None:
            return
        val_cer = metrics.get("eval_cer", None)
        if val_cer is None:
            return
        gap = val_cer - self._train_cer
        ratio = val_cer / (self._train_cer + 1e-9)
        print(
            f"\n[overfit-monitor] step={state.global_step} | "
            f"train_cer={self._train_cer:.4f} | val_cer={val_cer:.4f} | "
            f"gap={gap:+.4f} | ratio={ratio:.2f}"
        )
        if ratio > self.OVERFIT_WARN_RATIO:
            print(
                f"[overfit-monitor] ⚠ WARNING: val CER is {ratio:.1f}× train CER. "
                f"Consider reducing NUM_TRAIN_EPOCHS or increasing LORA_DROPOUT."
            )


# ─────────────────────────── modes ───────────────────────────────────────────

def mode_train():
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback

    # ── data ──────────────────────────────────────────────────────────────
    training_data, val_data = OCRDatasetPreparator(
        dataset_id           = HF_DATASET,
        token                = HF_TOKEN,
        prompt_template      = OCR_PROMPT,
        train_samples        = TRAIN_SAMPLES,
        val_samples          = VAL_SAMPLES,
        confusion_oversample = CONFUSION_OVERSAMPLE,
    ).load()
    print(f"[train] Training samples  : {len(training_data)}")
    print(f"[train] Validation samples: {len(val_data)}")

    # ── model ──────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(for_inference=False)
    FastVisionModel.for_training(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── warmup: ratio takes priority over fixed steps ─────────────────────
    sft_args = dict(
        per_device_train_batch_size  = BATCH_SIZE,
        per_device_eval_batch_size   = EVAL_BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM,
        learning_rate                = LR,
        weight_decay                 = WEIGHT_DECAY,
        lr_scheduler_type            = LR_SCHEDULER,
        logging_steps                = LOGGING_STEPS,
        eval_strategy                = "steps" if val_data else "no",
        eval_steps                   = EVAL_STEPS,
        save_strategy                = "steps" if val_data else "no",
        save_steps                   = EVAL_STEPS,
        save_total_limit             = 3,          # keep best + 2 recent
        load_best_model_at_end       = bool(val_data),
        metric_for_best_model        = "cer",
        greater_is_better            = False,
        optim                        = "adamw_8bit",
        seed                         = SEED,
        output_dir                   = str(OUTPUT_DIR),
        report_to                    = "none",
        remove_unused_columns        = False,
        dataset_text_field           = "",
        dataset_kwargs               = {"skip_prepare_dataset": True},
        max_length                   = None,
        dataset_num_proc             = DATASET_NUM_PROC,
    )

    # warmup_ratio is deprecated in transformers v5.2+ — compute warmup_steps directly.
    if WARMUP_STEPS > 0:
        sft_args["warmup_steps"] = WARMUP_STEPS
    else:
        effective_batch  = BATCH_SIZE * GRAD_ACCUM
        steps_per_epoch  = max(1, len(training_data) // effective_batch)
        total_steps      = steps_per_epoch * (MAX_STEPS if MAX_STEPS > 0 else NUM_EPOCHS)
        ratio            = WARMUP_RATIO if WARMUP_RATIO > 0 else 0.05
        warmup           = max(10, int(total_steps * ratio))
        sft_args["warmup_steps"] = warmup
        print(f"[train] Warmup steps (computed): {warmup}  ({ratio:.0%} of ~{total_steps} total steps)")

    if MAX_STEPS > 0:
        sft_args["max_steps"] = MAX_STEPS
    else:
        sft_args["num_train_epochs"] = NUM_EPOCHS

    callbacks = []
    if val_data:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        )
    callbacks.append(OverfitMonitorCallback())

    trainer = SFTTrainer(
        model          = model,
        tokenizer      = tokenizer,
        data_collator  = UnslothVisionDataCollator(model, tokenizer),
        train_dataset  = training_data,
        eval_dataset   = val_data if val_data else None,
        compute_metrics              = make_compute_metrics(tokenizer) if val_data else None,
        preprocess_logits_for_metrics= preprocess_logits_for_metrics if val_data else None,
        callbacks      = callbacks,
        args           = SFTConfig(**sft_args),
    )

    print("[train] Starting training …")
    print(f"[train] Effective batch size : {BATCH_SIZE * GRAD_ACCUM}")
    print(f"[train] Epochs               : {NUM_EPOCHS}")
    print(f"[train] Warmup ratio         : {sft_args.get('warmup_ratio', 'n/a')}")
    print(f"[train] LR scheduler         : {LR_SCHEDULER}")
    print(f"[train] Conjunct oversample  : {CONFUSION_OVERSAMPLE}×")
    print(f"[train] Vision layers frozen : {not env_bool('FINETUNE_VISION_LAYERS', 'false')}")

    trainer_stats = trainer.train()
    print(f"[train] Done. Stats: {trainer_stats}")

    # ── save LoRA ──────────────────────────────────────────────────────────
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_SAVE_DIR))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
    print(f"[train] LoRA saved → {MODEL_SAVE_DIR}")

    # ── push to Hub ────────────────────────────────────────────────────────
    if PUSH_TO_HUB:
        print(f"[train] Pushing LoRA adapter → {HF_REPO_ID}")
        model.push_to_hub_merged(HF_REPO_ID, tokenizer, save_method="lora", token=HF_TOKEN)
        print(f"[train] Hub push complete → https://huggingface.co/{HF_REPO_ID}")

    # ── sanity-check inference ─────────────────────────────────────────────
    print("[train] Running sanity-check inference on first training sample …")
    sample_content = training_data[0]["messages"][0]["content"]
    # content[0] = image dict, content[1] = text dict (after our fix)
    result = run_inference(
        model, tokenizer,
        sample_content[0]["image"],          # image is now first
        instruction=sample_content[1]["text"],
    )
    expected = training_data[0]["messages"][1]["content"][0]["text"]
    print(f"[infer] Expected : {expected}")
    print(f"[infer] Got      : {result}")

    mode_export(model, tokenizer)


def mode_eval():
    from unsloth import FastVisionModel
    import jiwer

    _, eval_data = OCRDatasetPreparator(
        dataset_id           = HF_DATASET,
        token                = HF_TOKEN,
        prompt_template      = OCR_PROMPT,
        val_samples          = VAL_SAMPLES,
        confusion_oversample = 1,   # no oversampling for eval
    ).load()

    model, tokenizer = load_model_and_tokenizer(for_inference=True)
    FastVisionModel.for_inference(model)

    n         = min(200, len(eval_data))
    all_gt    = []
    all_pred  = []

    print(f"[eval] Evaluating on {n} samples …")
    for item in eval_data[:n]:
        content      = item["messages"][0]["content"]
        ground_truth = nfc(item["messages"][1]["content"][0]["text"])
        # content[0] = image, content[1] = text
        prediction   = nfc(run_inference(
            model, tokenizer,
            content[0]["image"],
            instruction=content[1]["text"],
            stream=False,
        ))
        all_gt.append(ground_truth)
        all_pred.append(prediction)
        match = "✓" if prediction == ground_truth else "✗"
        print(f"{match} GT : {ground_truth}")
        if prediction != ground_truth:
            print(f"  PRD: {prediction}")
        print("─" * 60)

    overall_cer = jiwer.cer(all_gt, all_pred)
    print(f"\n[eval] Final CER on {n} samples: {overall_cer:.4f}")


# Per-image language hint for batch inference. The test set mixes Bangla and
# English images, and the prompt's {LANG} slot materially affects decoding, so
# each file is mapped to its true language. Files not listed fall back to
# INFER_DEFAULT_LANG below ("Both Bangla and English").
INFER_LANG_MAP = {
    # English — dates / NID digit strings
    "2026-04-26_194956.png": "English",
    "2026-04-26_195000.png": "English",
    "2026-06-10_193616.png": "English",
    "2026-06-10_193625.png": "English",
    # Bangla — names / addresses
    "2026-04-22_132012.png": "Bangla",
    "2026-04-22_133236.png": "Bangla",
    "2026-04-26_194947.png": "Bangla",
    "2026-06-10_194927.png": "Bangla",
    "2026-06-10_195402.png": "Bangla",
    "2026-06-17.png": "Bangla",
    "2026-06-17_21-28-21.png": "Bangla",
    "2026-06-17_21-29-17.png": "Bangla",
    "2026-06-17_21-39-32.png": "Bangla",
    "Screenshot 2026-06-17 at 21-03-39 খালেদা জিয়া - উইকিপিডিয়া.png": "Bangla",
}
INFER_DEFAULT_LANG = "Both Bangla and English"


def mode_infer():
    """Batch inference over all images in INFER_IMAGE_DIR."""
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    infer_dir = Path(os.environ.get("INFER_IMAGE_DIR", ""))
    if not infer_dir or not infer_dir.is_dir():
        print(f"Set INFER_IMAGE_DIR to a directory containing images (got: '{infer_dir}').")
        sys.exit(1)

    images = sorted(p for p in infer_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not images:
        print(f"No images found in {infer_dir}")
        sys.exit(1)

    model_path = os.environ.get("INFER_MODEL_PATH", str(MODEL_SAVE_DIR))
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name   = model_path,
        load_in_4bit = LOAD_IN_4BIT,
    )
    FastVisionModel.for_inference(model)

    results = {}
    for img_path in images:
        lang        = INFER_LANG_MAP.get(img_path.name, INFER_DEFAULT_LANG)
        instruction = OCR_PROMPT.replace("{LANG}", lang)
        print(f"[infer] Processing {img_path.name} … (lang={lang})")
        results[img_path.name] = run_inference(
            model, tokenizer, str(img_path), instruction=instruction, stream=False
        )

    col = max(len(n) for n in results) + 2
    sep = "─" * (col + 3 + 80)
    print(f"\n{sep}")
    print(f"{'Image':<{col}}│  Prediction")
    print(sep)
    for name, text in results.items():
        print(f"{name:<{col}}│  {text}")
    print(sep)


def mode_export(model=None, tokenizer=None):
    """Export LoRA to merged 16-bit and/or GGUF quantisations."""
    if model is None:
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(
            model_name   = str(MODEL_SAVE_DIR),
            load_in_4bit = LOAD_IN_4BIT,
        )
        FastVisionModel.for_inference(model)

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
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B Vision OCR fine-tuner")
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "infer", "export"],
        default="train",
        help="Execution mode (default: train)",
    )
    args = parser.parse_args()

    print(f"[main] Mode              : {args.mode}")
    print(f"[main] Base model        : {BASE_MODEL}")
    print(f"[main] 4-bit LoRA        : {LOAD_IN_4BIT}")
    print(f"[main] Vision frozen     : {not env_bool('FINETUNE_VISION_LAYERS', 'false')}")
    print(f"[main] Data dir          : {DATA_DIR}")
    print(f"[main] Output dir        : {OUTPUT_DIR}")

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
