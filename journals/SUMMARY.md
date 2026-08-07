# ORENA FOCUS — SFT Project Summary

Synthesis of the 2026-07-16 → 2026-07-20 work: building a supervised
fine-tuning + evaluation pipeline for the FOCUS surgical-VQA frame track, and
running a cross-model / cross-dataset generalization study.

Detailed daily logs: [2026-07-16](2026-07-16.md) · [2026-07-17](2026-07-17.md) ·
[2026-07-20](2026-07-20.md) · [2026-07-21](2026-07-21.md) ·
pipeline reference: [sft_pipeline.md](sft_pipeline.md) ·
data reference: [heico_data_structure.md](heico_data_structure.md)

---

## 1. The data

Two layers, often confused:

- **Raw visual data** (local, `/projects/datasets_ML/orena/<dataset>/`):
  `videos/`, `frames/`, `overlayed/`, plus `train/val/test.lance` — flat
  columnar tables of **individual JPEG frames** (one row per frame, not whole
  videos), split at the video level. heico: 30 videos / 8.68M frames.
- **QA benchmark** (Hugging Face, `orena-dkfz/{heico,lapchole}-focus-vqa`):
  loaded via `focus.FocusDataset`. Two tracks — `frame` (single still image)
  and `segment` (video clip). We use **frame track only**.

**Datasets are different procedures** — this is the crux of the whole study:
- **heico** = Proctocolectomy / Rectal / Sigmoid resection → *lower abdomen, colorectal*
- **lapchole** = laparoscopic cholecystectomy → *upper abdomen, gallbladder*

Taxonomy: 15 leaf capabilities in 5 groups; frame track only exercises ~5 of
them (static-image properties). `ood` and `clinical` flags are **False for every
question in both datasets**, so the taxonomy's own OOD dimension is unusable —
hence using heico→lapchole transfer as the OOD proxy.

## 2. The pipeline (`orena_sft/`)

| Script | Purpose |
|---|---|
| `build_frame_sft_dataset.py` | HF QA → chat-format JSONL (train/eval/test) |
| `sft_train_qwen_frame.py` / `.slurm` | LoRA SFT for Qwen3.5 |
| `sft_train_gemma_frame.py` / `.slurm` | LoRA SFT for Gemma-4 |
| `evaluate_qwen_frame.py` / `.slurm` | Eval fine-tuned Qwen |
| `evaluate_gemma_frame.py` / `.slurm` | Eval fine-tuned Gemma |
| `base_model_eval.py` / `.slurm` | Eval base Qwen (forgetting baseline) |
| `gemma_eval_base.py` / `.slurm` | Eval base Gemma (follows Google's guidance) |

Key design decisions:
- **Images referenced by path, never copied** — frames already exist on disk;
  embedding them would have produced GB-scale exports (the mistake visible in
  `probing_export/*.arrow`).
- **Eval split at the video level**, stratified by procedure — question-level
  splitting would leak correlated frames from the same video.
- **Loss masked to the answer only** — verified: of 552 tokens in one example,
  546 (image + question + template) masked, 6 (`'Clip, Silicone loop<|im_end|>'`)
  are the target.
- **Scoring uses the FOCUS library's own `Evaluator`** — deterministic parsing
  for closed formats (`binary`/`number`/`fo_class`/`time`), LLM-as-judge for
  `open_ended`/`multiple_choice`, with video-level bootstrapped CIs.

## 3. Results

All numbers are `overall MEAN` from `summary.csv`. **situs** =
`spatial_localization_situs` (the knowledge-heavy anatomy capability, n=25 on
lapchole — small, treat with care).

### Fine-tuned on heico → evaluated on lapchole (OOD)

| Model | LoRA | overall | situs | fo_class |
|---|---|---|---|---|
| Qwen3.5-9B | r=8 | 40.5% | **17.9%** | 36.5% |
| Qwen3.5-9B | r=16 | 41.0% | **0.0%** | 38.6% |
| Qwen3.5-9B | r=32 | 42.0% | **1.2%** | 40.4% |
| Qwen3.5-27B | r=16 | 42.1% | 2.4% | 37.8% |
| Gemma-4-31B | r=16 | 39.4% | **22.6%** | 34.8% |

### Base models (no fine-tuning) → lapchole

| Model | overall | situs | fo_class |
|---|---|---|---|
| Qwen3.5-9B | 18.1% | **56.0%** | 0.2% |
| Qwen3.5-27B | 15.2% | 36.9% | 0.3% |
| Gemma-4-31B | *(4.0%, `--limit 50` only — not comparable)* | — | 0.0% |

### In-distribution (heico test)

| Model | overall | situs |
|---|---|---|
| Qwen3.5-9B (combined data, ckpt-600) | 62.3% | 63.7% |
| Qwen3.5-27B heico-only | 61.4% | 70.9% |
| Qwen3.5-27B base | 18.6% | 8.3% |

## 4. Key findings

### (a) SFT trades knowledge for format compliance

The single clearest result. Base models score ~15–18% overall but **~0% on
`fo_class`** — not from blindness, but because they answer in prose ("The
surgical foreign object visible is a surgical clip") instead of the required
terse format, so the deterministic parser rejects them. SFT's main contribution
is teaching **output format**, worth ~+25 points overall.

Meanwhile the same SFT **destroys pretrained domain knowledge**: `situs` drops
56.0% → 0.0% (Qwen-9B r16). The base model knows cholecystectomy anatomy
(`Gallbladder`, `cystic duct`) from pretraining; heico-only fine-tuning
overwrites it with colorectal priors (`Rectum`, `Lower left quadrant`) and it
answers with the wrong procedure's vocabulary. **Catastrophic forgetting,
directly measured.**

### (b) Lower LoRA rank forgets less

The rank sweep supports the capacity/forgetting hypothesis:

| rank | overall | situs |
|---|---|---|
| r=8 | 40.5% | **17.9%** |
| r=16 | 41.0% | 0.0% |
| r=32 | 42.0% | 1.2% |

r=8 retains meaningfully more anatomy knowledge at ~equal overall accuracy —
less adapter capacity means less capacity to overwrite. (Single runs, no seed
repeats; r=16 vs r=32 ordering is within noise.)

### (c) Model scale buys almost nothing here

9B → 27B → 31B moves overall accuracy 41.0% → 42.1% → 39.4%. **Parameters are
not the bottleneck**; data diversity and format alignment dominate.

### (d) Distribution shift costs ~19 points

Qwen-27B heico-only: **61.4% in-distribution (heico) → 42.1% OOD (lapchole)**.
And the gap is concentrated in knowledge-heavy capabilities: situs 70.9% → 2.4%.

### (e) Overfitting starts around epoch ~1.5–1.8

Consistent across models: eval loss bottoms out then climbs. Gemma stopped early
at step 350/675 (best=200, epoch 1.56); Qwen heico-only best was step 400
(epoch 1.78) of 675. Three epochs is too many for this dataset size.

## 5. Bugs found & lessons

| Bug | Lesson |
|---|---|
| `mm_token_type_ids` missing → forward crash | Multimodal models need per-token modality IDs; a hand-rolled collator must propagate every processor output. |
| `AutoProcessor` silently degraded to a bare tokenizer on intermediate checkpoints | `Trainer` saves whatever `processing_class` is; pass the **full processor**, not `.tokenizer`. |
| Three runs writing to one checkpoint dir (r=16/r=32/r=8 mixed) | Fixed `--run-name` + concurrent jobs = silent corruption. Distinct names per experiment; a collision guard is still unimplemented. |
| `load_best_model_at_end` picked the wrong checkpoint | `--save-steps` must equal `--eval-steps`, or the numerically best eval step has no saved weights. |
| Early stopping never fired (patience 4, 7 evals total) | Patience must fit inside the run: `eval_steps` small enough that patience is reachable. |
| LoRA crash on `Gemma4ClippableLinear` | Gemma's vision tower reuses `q_proj`-style names but wraps them in a custom class; scope LoRA to `language_model` by regex. (Qwen was implicitly text-only by naming accident.) |
| **`enable_thinking` misread** | I judged a rendered prompt from its **last 45 characters** and concluded the flag was "inverted". It wasn't: `<\|channel>thought\n<channel\|>` is an *empty closed* thought block (= thinking OFF), and `enable_thinking=True` injects a `<\|think\|>` **system** prompt. **Always print the whole rendered string.** |

The `enable_thinking` error was caught by the user from a training log showing
the model emitting `thought\nThinking Process:...`. The corrected approach follows
Google's official guidance: `enable_thinking=False` + `processor.parse_response()`,
which returns `{"thinking": ..., "content": ...}` — no hand-rolled stripping.

## 6. Open items

- **Fair base-Gemma baseline**: rerun full (not `--limit 50`) with the corrected
  `enable_thinking=False` + `parse_response` script. The earlier 6.1% was
  produced by the buggy prompt (100% of predictions carried a `thought` prefix,
  30% truncated mid-thought) and is not a valid measurement.
- **Align `evaluate_gemma_frame.py`** (fine-tuned eval) with the same corrected
  pattern — it still uses the hand-built prompt.
- **Gemma in-distribution (heico) eval** not yet run.
- **Train/inference format mismatch (Gemma)**: training targets have no thought
  block, but Google's inference prompt pre-fills an empty one. Worth aligning in
  the next training run.
- **Collision guard**: make training refuse to start if `--output-dir` already
  contains a `trainer_state.json`.

## 7. Ideas for improving OOD generalization

Ranked by expected value given the findings above:

1. **Add domain diversity** — the most reliable lever. Public cholecystectomy
   VQA data exists (SSG-VQA ~960k Qs on CholecT45; CholeVidQA-32K; Cholec80 /
   CholecT50 annotations convertible to this format). Adding cholecystectomy
   data would move lapchole from OOD to in-distribution.
2. **Anti-forgetting**: lower LoRA rank (supported by the sweep), early stopping
   on an **OOD** dev split rather than in-distribution loss, KL-anchoring to the
   base model.
3. **Put the label space in the prompt** — `multiple_choice` transferred nearly
   perfectly (80.5% vs 81.9% in-distribution) *because the options are in the
   prompt*. Listing valid FO classes / anatomical structures converts open
   generation into constrained selection.
4. **Prompt-template diversification** — FOCUS questions are highly templated;
   recent work attributes much of SFT's apparent OOD failure to frozen-prompt
   memorization, recoverable by paraphrasing during training.
5. **SFT → GRPO** — most FOCUS formats are rule-verifiable, so `Evaluator` is
   nearly a ready-made reward function; literature reports RL "heals" the OOD
   forgetting SFT causes. Expensive; do after the cheaper levers.
