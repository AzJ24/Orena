# Method — FOCUS frame track SFT

Reference for [sft_train_qwen_frame_ddp.py](sft_train_qwen_frame_ddp.py) and its
launcher [sft_train_qwen_frame_ddp.slurm](sft_train_qwen_frame_ddp.slurm):
what data goes in, what model is trained, and how.

---

## 1. Data

### Source and construction

Built by [build_frame_sft_dataset.py](build_frame_sft_dataset.py) from the FOCUS
challenge data, loaded straight from HuggingFace via `focus.FocusDataset` (not
from any QA cache). Only the **frame track** is used: one still image per
question (`duration == 0`), so no video decoding happens at train time.

Two source datasets are merged into the `combined` export:

- **heico** — Proctocolectomy, Rectal Resection (train), Sigmoid Resection (test)
- **lapchole** — Laparoscopic Cholecystectomy

Images are never copied or embedded. Each record stores the path of the
already-extracted frame JPEG under `<root_dir>/<dataset>/frames/`, resolved with
the same formula `FocusFrameDataset` uses internally
(`frame_idx = round(start_time * base_fps)`). Rows whose frame is missing on disk
are dropped. The JSONL exports stay at KB scale.

### Splits

`eval` is carved out of the HF **train** split at the **video** level (many
frame-track questions from one video are correlated, so a question-level split
would leak), stratified by `procedure_type`, `eval_frac = 0.12`, `seed = 42`. The
official HF **test** split is left untouched as the held-out benchmark.

| split | rows | videos | heico / lapchole |
|---|---|---|---|
| `combined/train.jsonl` | 12,225 | 81 | 7,200 / 5,025 |
| `combined/eval.jsonl` | 1,523 | 11 | 800 / 723 |
| `combined/test.jsonl` | 6,252 | 38 | 4,000 / 2,252 |
| `combined_all/train.jsonl` | 20,000 | 130 | train+eval+test pooled |

The test split is a **procedure shift**: its heico half is Sigmoid Resection,
a procedure type that appears in neither train nor eval.

### Question mix (train split)

| answer format | rows | | primary capability | rows |
|---|---|---|---|---|
| `fo_class` | 5,587 | | OBJECT_IDENTIFICATION | 5,217 |
| `number` | 3,802 | | OBJECT_AGGREGATION | 4,926 |
| `open_ended` | 1,170 | | SPATIAL_LOCALIZATION_CAMERA | 1,397 |
| `binary` | 1,094 | | OBJECT_ATTRIBUTES | 423 |
| `multiple_choice` | 572 | | SPATIAL_LOCALIZATION_SITUS | 260 |

### Record format

Chat-format JSONL, one image and one question per record, plus metadata used for
splitting and later analysis (`qID`, `source_dataset`, `videoID`,
`procedure_type`, `primary_capability`, `secondary_capabilities`, `format`):

```json
{"messages": [
  {"role": "user", "content": [{"type": "image", "image": "/…/frame0430775.jpg"},
                                {"type": "text",  "text": "Which combination of foreign object classes is visible in this frame? …"}]},
  {"role": "assistant", "content": [{"type": "text", "text": "Clip, Silicone loop"}]}
]}
```

**Targets are bare answers** — no reasoning, no sentence. This is the single
constraint that drives the prompting design below.

---

## 2. Model

- **Base:** `Qwen/Qwen3.5-9B` (`Qwen3_5ForConditionalGeneration`), a multimodal
  VLM, loaded in **bfloat16**.
- **Adaptation:** **LoRA** (PEFT) by default — `r = 8`, `alpha = 16`,
  `dropout = 0.05`, `task_type = CAUSAL_LM`, applied to all attention and MLP
  projections (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
  The base LM and the vision tower stay frozen. `--no-lora` switches to a full
  fine-tune.
- **Processor:** `AutoProcessor` for the same model id, saved alongside every
  checkpoint (`processing_class=processor`, not just the tokenizer — otherwise
  intermediate checkpoints lack `preprocessor_config.json` and later
  `AutoProcessor.from_pretrained()` silently degrades to a bare tokenizer).

---

## 3. Method

### 3.1 Supervised fine-tuning objective

TRL `SFTTrainer` with a **custom collator** ([sft_train_qwen_frame_ddp.py:57](sft_train_qwen_frame_ddp.py#L57)).
Standard next-token cross-entropy, but **loss is computed on the answer only**:

1. Render the full chat with `apply_chat_template(tokenize=False)`.
2. Find the literal assistant marker `<|im_start|>assistant\n<think>\n\n</think>\n\n`
   inside that rendered string (`rindex`) and slice the prefix.
3. Tokenize prefix and full text; set `labels[:prompt_len] = -100`.

The marker is located *inside the rendered text* rather than re-deriving the
prompt length from a separate `add_generation_prompt=True` call, because Qwen3.5
renders an already-answered turn (empty `<think>` block) differently from a fresh
generation prompt (thinking left open) — the two would disagree on the boundary.
Slicing a literal prefix guarantees the tokenized prompt is an exact prefix of the
full tokenization.

The collator also pads manually and builds `mm_token_type_ids` (0 = text,
1 = image, per Qwen3.5's M-RoPE convention; padding is typed 0), plus concatenated
`pixel_values` / `image_grid_thw`.

### 3.2 System prompt arms

A system prompt is prepended to **every** example, identically in the collator,
the eval-sample callback and the startup preview (`with_system()`), so training
and every generation path see the same thing. Four mutually exclusive settings:

| `--prompt-style` | content |
|---|---|
| `plain` (default) | no system prompt |
| `direct` | format-control prompt from [prompts.py](prompts.py): FO class vocabulary + answer-shape rules, bare answer requested |
| `structured` | REASONING line + ANSWER line — **rejected at argparse for training** |
| `--system-prompt-file` | a verbatim prompt file (e.g. a GEPA-evolved `best_prompt.txt`), overrides the above |

`structured` is refused loudly: the targets are bare answers, so training on that
pair would teach the model to *disobey* the prompt it is given. Supporting it
needs reasoning traces in the targets. `--fo-definitions` swaps the class-name
list for the full per-class descriptions (~700 extra prompt tokens/question).

The prompt itself is deliberately format-agnostic and OOD-safe: it keys off the
question's own wording rather than a format field (which does not exist at
submission time), names no anatomy/procedure/dataset, and reads the FO class
vocabulary live from `FOType.names()`.

`--condition-procedure` optionally prepends `Procedure type: <name>.` to each
question at load time. **Eval must use the same prompt settings as training** —
otherwise the numbers are a train/inference mismatch.

### 3.3 Distributed training (DDP)

Data-parallel across **2× H200 on one node**, launched by `torchrun
--standalone --nproc_per_node=2`. Each process holds a *full* 9B + LoRA replica on
its own GPU (`device_map={"": LOCAL_RANK}`, falling back to `"auto"` for a plain
single-GPU `python` launch) and processes a data shard; gradients are all-reduced
every step. Model-parallel (`device_map="auto"`) is incompatible with this and is
only the single-GPU path. Rationale and checklist: [ddp.md](ddp.md).

Three DDP-specific guards:

- `ddp_find_unused_parameters=False` — LoRA freezes the base and vision tower, so
  many parameters receive no gradient.
- The sample-generation callback and the startup preview run on **rank 0 only**
  and unwrap the DDP module before `.generate()`; a forward-only pass DDP does not
  expect would otherwise deadlock the other rank.
- wandb is disabled on non-zero ranks (`WANDB_MODE=disabled`).

`NCCL_IB_DISABLE=1` (single-node NVLink/PCIe, avoids InfiniBand probing stalls)
and `TOKENIZERS_PARALLELISM=false`.

**Effective batch is kept at 32** to match the single-GPU runs so the convergence
estimate carries over: with 2 GPUs, `--batch-size 16 --grad-accum 1`. DDP is never
bit-identical to single-GPU (different FP reduction order, `DistributedSampler`
sharding, per-rank dropout RNG) — matched effective batch buys *quality
equivalence*, like a different seed.

### 3.4 Hyperparameters and schedule

| | default | typical run |
|---|---|---|
| learning rate | `1e-4` | `1e-4` |
| epochs | `1.0` | `1.5` |
| per-device batch | `1` | `16` (×2 GPUs = 32) |
| grad accum | `16` | `1` |
| precision | bf16 | bf16 |
| seed | `42` | `42` |
| eval / save every | 200 steps | 150 steps |

`--max-steps` caps optimizer steps for smoke runs. `save_total_limit=20`.
`--early-stopping-patience` enables `EarlyStoppingCallback` on `eval_loss` with
`load_best_model_at_end`; argparse enforces `save_steps % eval_steps == 0` so a
checkpoint exists at every eval point.

### 3.5 Monitoring

- **wandb** (`orena-frame-sft`), initialized explicitly before the Trainer so the
  run carries a real name and a full config snapshot (model, LoRA rank, prompt
  style, dataset sizes, effective batch, hostname, SLURM job id) — `report_to="wandb"`
  alone would only produce a default `huggingface` project with a generated name.
- **Startup preview** ([sft_train_qwen_frame_ddp.py:207](sft_train_qwen_frame_ddp.py#L207)):
  prints one eval example before training — the full rendered chat, the
  ground-truth target, the inference-time generation prompt, and a live greedy
  generation from the untrained adapter.
- **`SampleGenerationCallback`**: after every eval, greedy-decodes one fixed eval
  example and appends `(question, gt_answer, pred_answer, image)` to
  `eval_samples.jsonl` and wandb — qualitative progress alongside the loss curve.

At the end, `trainer.save_model()` + `processor.save_pretrained()` write to
`--output-dir` (defaults to `orena_sft/checkpoints/<run_name>`).

---

## 4. How to run

```bash
sbatch --export=ALL,\
TRAIN_FILE=/home/ajenane/orena/orena_sft/sft_export/combined_all/train.jsonl,\
EVAL_FILE=/home/ajenane/orena/orena_sft/sft_export/combined/eval.jsonl,\
EXTRA_ARGS="--model-id Qwen/Qwen3.5-9B --prompt-style direct \
  --run-name combined-all-9b-8r-direct-ddp --lora-r 8 --lora-alpha 16 \
  --epochs 1.5 --batch-size 16 --grad-accum 1 --save-steps 150 --eval-steps 150" \
  orena_sft/sft_train_qwen_frame_ddp.slurm
```

SLURM request: `gpu-large`, `gpu:h200:2`, 1 node, 32 CPUs, 256 GB, 23 h.
Steps/epoch = `n_examples / effective_batch` (20,000 / 32 ≈ 625).
