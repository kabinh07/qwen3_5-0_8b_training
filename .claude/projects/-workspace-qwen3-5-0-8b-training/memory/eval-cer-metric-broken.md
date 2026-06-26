---
name: eval-cer-metric-broken
description: train.py eval CER is inflated (teacher-forced full-sequence argmax) and triggers premature early stopping
metadata:
  type: project
---

In `train.py`, `make_compute_metrics` (~line 374) computes eval CER from `preprocess_logits_for_metrics` (line 402) which returns `logits.argmax` over the WHOLE sequence. Labels are target-only (prompt positions masked to -100→pad), so predictions are far longer than labels → CER >1.0 (pure insertions). Reported eval CER (~0.95 overall, 1.2–1.5 English) does NOT reflect real autoregressive quality — actual `--mode infer` output is mostly clean.

**Why:** This is teacher-forced next-token argmax over prompt+answer compared against answer-only. It is not generation accuracy.

**How to apply:** Don't trust eval_cer in trainer_state.json. It also drives `EarlyStoppingCallback` (patience 3): the p4/outputs_8 run stopped at step 3000 = epoch 1.04 of a planned 5 epochs because this broken metric "stopped improving." Before any retrain (e.g. on p5 dataset), either slice predictions to the label region in compute_metrics, or disable CER-based early stopping, so training runs the full schedule. Confirmed failure modes from the synthetic_data_brief.txt still reproduce in outputs_8: ড/দ (গেন্ডারিয়া→গেন্দারিয়া), repeated-digit hallucination (extra 0 in NID zero-run), low-ink misreads.
