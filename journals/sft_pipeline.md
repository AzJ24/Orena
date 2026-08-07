# SFT Pipeline — `orena_sft/`

Documents the full pipeline for supervised fine-tuning Qwen3.5-9B on the
FOCUS frame track (single-image VQA), covering both scripts, the design
choices behind them, and how to launch a run.

## What it does, end to end

```
build_frame_sft_dataset.py  →  sft_export/combined/{train,eval,test}.jsonl  →  sft_train_qwen_frame.py  →  checkpoints/<run_name>/
```

1. **`build_frame_sft_dataset.py`** loads the FOCUS frame-track QA pairs
   directly from Hugging Face (not from any local cache), resolves each
   question to its single existing frame JPEG on disk, splits off an eval
   set at the video level, and writes three chat-formatted JSONL files.
2. **`sft_train_qwen_frame.py`** consumes those JSONL files and fine-tunes
   Qwen3.5-9B with LoRA via TRL's `SFTTrainer`, logging to Weights & Biases
   and periodically writing out a qualitative sample prediction.
3. **`sft_train_qwen_frame.slurm`** is the Slurm submission wrapper — GPU
   access on this cluster is only available via Slurm (`gpu-large` /
   `gpu-large-interactive` partitions on node `gpu38`, either H200s or
   RTX PRO 6000s), so training never runs directly on a login/interactive
   node.

## `build_frame_sft_dataset.py`

### What it includes
- Loads QA pairs for the **frame track only** (single still image per
  question, `duration == 0`) via `focus.FocusDataset`, for one or more
  datasets (`heico`, `lapchole`).
- Resolves each question to its frame JPEG using the exact same formula
  `FocusFrameDataset` uses internally (`round(start_time * base_fps)`),
  reading from `<root_dir>/<dataset>/frames/<video_stem>/frame{idx:07d}.jpg`.
- Splits `train` into `train`/`eval` **at the video level**, stratified by
  `procedure_type`, default `--eval-frac 0.12`.
- Writes `train.jsonl`, `eval.jsonl`, `test.jsonl` as chat-formatted records.

### Design choices
- **No caching layer.** Unlike `scripts/build_probing_dataset.py`, this does
  not read from `webapp/cache/qa_cache.parquet` — it hits HF/its local cache
  directly, same as `data_exploration.ipynb`.
- **Images are never copied or embedded.** Each record stores a file path
  string, not pixel bytes. The frames already exist on disk (extracted
  earlier via `FrameExtractorPreprocessor`); duplicating them into the
  export would blow up disk usage for no benefit — this is the same lesson
  learned from `probing_export/*.arrow` files elsewhere in this repo, which
  embed JPEG bytes and are 100s of MB each. This export instead stays at
  KB/MB scale.
- **Eval split is per-video, not per-question.** Many frame-track questions
  from the same video are highly correlated (same background, same object
  instances), so a row-level random split would leak information between
  train and eval. Splitting whole videos out avoids that.
- **`test.jsonl` is never touched by training.** It's the official HF test
  split, kept as the final held-out benchmark — the training script doesn't
  even accept it as an argument.
- **Multi-dataset merge qualifies `videoID` per-dataset** (`"heico/0009 -
  Heico - Prokto - 10.avi"`) so the eval split never accidentally treats
  same-named videos from different source datasets as one video. Each
  record also keeps a `source_dataset` field for later filtering/analysis.
- **Location-independent defaults.** `--out-dir` defaults to
  `<script-dir>/sft_export`, resolved via `Path(__file__).resolve().parent`
  rather than cwd, so it works regardless of where you invoke it from
  (this mattered after the whole pipeline was moved into `orena_sft/`).

### Launch command
```bash
.venv/bin/python orena_sft/build_frame_sft_dataset.py --datasets heico lapchole
```

### Args
| Flag | Default | Meaning |
|---|---|---|
| `--datasets` | `heico` | One or more of `heico`, `lapchole`; merges them into one export. |
| `--root-dir` | `/projects/datasets_ML/orena/` | Root containing `<dataset>/frames/` for each dataset. |
| `--out-dir` | `orena_sft/sft_export` | Where `train.jsonl`/`eval.jsonl`/`test.jsonl` are written. |
| `--eval-frac` | `0.12` | Fraction of train videos (per dataset, per procedure type) held out for eval. |
| `--seed` | `42` | RNG seed for the eval video split. |

### Verified output (heico + lapchole combined)
```
Loading 'heico' frame track...     train: 8000   test: 4000
Loading 'lapchole' frame track...  train: 5748   test: 2252

Video-level eval split: 11 eval videos, 81 train videos (target eval_frac=0.12)
  final train: 12225 rows
  eval:        1523 rows
  test:        6252 rows (untouched HF test split)
```
All rows resolved to existing frames on disk — 0 dropped for either dataset.

### Record format (one line of `train.jsonl`)
```json
{
  "qID": "8455",
  "source_dataset": "heico",
  "videoID": "heico/0009 - Heico - Prokto - 10.avi",
  "procedure_type": "Proctocolectomy",
  "primary_capability": "OBJECT_IDENTIFICATION",
  "secondary_capabilities": ["OBJECT_AGGREGATION"],
  "format": "fo_class",
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "/projects/datasets_ML/orena/heico/frames/0009 - Heico - Prokto - 10/frame0430775.jpg"},
      {"type": "text", "text": "Which combination of foreign object classes is visible in this frame? Please provide the class names or answer with none."}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "Clip, Silicone loop"}]}
  ]
}
```

## `sft_train_qwen_frame.py`

### What it includes
- Loads `Qwen3.5-9B` (`Qwen3_5ForConditionalGeneration`) in bf16, wraps it
  in a LoRA adapter by default (`peft.get_peft_model`).
- A **custom collate function** that builds full chat sequences, masks the
  loss to the answer only, and supplies Qwen3.5's M-RoPE-required
  `mm_token_type_ids` field.
- **`SampleGenerationCallback`**: after every eval, runs real generation on
  one fixed eval example and logs `(image, question, gt_answer,
  pred_answer)` to a local JSONL and to W&B.
- **Early stopping** (optional): `EarlyStoppingCallback` on `eval_loss` with
  `load_best_model_at_end=True`.
- **W&B logging** with a real project name, a timestamped run name, and a
  full hyperparameter/dataset-size config snapshot.

### Design choices

**Loss masking.** The Qwen3.5 chat template always renders an
already-answered assistant turn as
`<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}` — with
an *empty but present* think block whenever no `reasoning_content` is
supplied (always true for this QA data). That differs from what
`add_generation_prompt=True` renders for a fresh turn (`enable_thinking`
defaults to `True`, leaving `<think>\n` **open**, not closed). Because those
two renderings diverge, re-deriving the prompt/answer boundary from a
separate `add_generation_prompt=True` call would give a length that doesn't
actually match the full sequence's tokenization. Instead, the collator finds
the literal `ASSISTANT_MARKER` string
(`"<|im_start|>assistant\n<think>\n\n</think>\n\n"`) inside the full
rendered text and slices there — guaranteed to be an exact string prefix, so
prompt-only tokenization is guaranteed to match the corresponding prefix of
the full tokenization. Verified empirically: of 552 total tokens in a real
example, 546 (image + question + template scaffolding) are masked
(`labels = -100`), leaving only `'Clip, Silicone loop<|im_end|>\n'` (6
tokens) as the actual training target.

**`mm_token_type_ids`.** Qwen3.5 uses multimodal rotary position embeddings
(M-RoPE), which require a per-token tensor marking text (`0`) vs. image
(`1`) positions. The processor returns this automatically alongside
`input_ids`, but a hand-rolled collator has to remember to propagate it —
omitting it raises `ValueError: Multimodal data was passed... but
mm_token_type_ids is missing`. Padding positions are typed `0` (text), same
as any other filler.

**All QA pairs per frame are preserved as separate examples.** 1,322 of
5,372 unique train frames (heico) have more than one QA pair; each is kept
as its own self-contained `(image, question, answer)` record rather than
deduplicated or merged into a multi-turn conversation.

**No custom image embedding/caching.** Every training step opens the image
fresh via `PIL.Image.open(image_path)` and runs it through the processor
twice per example (once for the full sequence, once for the prompt-only
prefix used to compute `prompt_len`) — simpler and more obviously correct
than trying to cache/reuse image tensors, at the cost of some redundant
CPU-side preprocessing.

**W&B project/run naming.** `report_to="wandb"` alone dumps every run into
the default `"huggingface"` project with an auto-generated name (e.g.
`astral-firefly-135`) — not useful for tracking multiple experiments. The
script instead calls `wandb.init()` explicitly *before* constructing
`SFTTrainer`, with a real `--wandb-project` (default `orena-frame-sft`) and
a timestamped `run_name` (`{model}-{lora|full}-{timestamp}`, e.g.
`Qwen3.5-9B-lora-2026-07-16_16-41-57`). `SFTTrainer`'s `WandbCallback`
detects the already-active run and reuses it instead of starting its own.

**`--output-dir` defaults to the run name.** So checkpoints land in
`orena_sft/checkpoints/<run_name>/`, matching the W&B run exactly with no
extra bookkeeping — you can always find a checkpoint from a W&B run's name
alone.

**Early stopping requires aligned eval/save cadence.** `load_best_model_at_end`
needs a saved checkpoint at every eval point, so the script errors out early
(`ap.error(...)`) if `--early-stopping-patience` is set but `--save-steps`
isn't a multiple of `--eval-steps` — rather than letting `Trainer` fail
confusingly later.

**`--max-steps` for calibration.** Since there was no empirical throughput
number for this model/hardware combination, `--max-steps` (default `-1`,
disabled) lets you cap a run at e.g. 20 steps to read real `it/s` off the
Trainer logs before committing to a multi-hour job.

### Example generation prompt (what the model actually sees)

User turn (image tokens abbreviated):
```
<|im_start|>user
<|vision_start|><|image_pad|>...×510...<|image_pad|><|vision_end|>Which combination of foreign object classes is visible in this frame? Please provide the class names or answer with none.<|im_end|>
<|im_start|>assistant
<think>

</think>

```
Training target (unmasked labels only):
```
Clip, Silicone loop<|im_end|>
```
(510 image-patch tokens for a single 224×224-ish frame, in this measured
example; total sequence 552 tokens, of which only the final 6 carry loss.)

### Launch command
```bash
sbatch --export=ALL,EXTRA_ARGS="--epochs 3 --batch-size 16 --grad-accum 2 --eval-steps 100 --save-steps 200 --early-stopping-patience 4" \
    orena_sft/sft_train_qwen_frame.slurm
```
Or directly (inside an interactive Slurm allocation — GPUs aren't reachable
outside Slurm on this cluster):
```bash
srun -p gpu-large-interactive --gres=gpu:rtx_pro_6000:1 --pty bash
.venv/bin/python orena_sft/sft_train_qwen_frame.py --epochs 3 --batch-size 16 --grad-accum 2 \
    --eval-steps 100 --save-steps 200 --early-stopping-patience 4
```

### Args
| Flag | Default | Meaning |
|---|---|---|
| `--train-file` | `orena_sft/sft_export/combined/train.jsonl` | Training JSONL. |
| `--eval-file` | `orena_sft/sft_export/combined/eval.jsonl` | Eval JSONL. |
| `--output-dir` | `orena_sft/checkpoints/<run_name>` | Checkpoint output dir; auto-named after the W&B run if not set. |
| `--model-id` | `Qwen/Qwen3.5-9B` | HF model id. |
| `--epochs` | `1.0` | Training epochs (ignored if `--max-steps > 0`). |
| `--max-steps` | `-1` (disabled) | Caps total optimizer steps; use for calibration/smoke runs. |
| `--batch-size` | `1` | Per-device batch size. |
| `--grad-accum` | `16` | Gradient accumulation steps. Effective batch = `batch_size × grad_accum`. |
| `--lr` | `1e-4` | Learning rate. |
| `--no-lora` | off | Full fine-tune instead of LoRA. |
| `--lora-r` | `16` | LoRA rank. |
| `--lora-alpha` | `32` | LoRA alpha. |
| `--eval-steps` | `200` | Eval every N optimizer steps. |
| `--save-steps` | `200` | Checkpoint every N optimizer steps; must be a multiple of `--eval-steps` if early stopping is enabled. |
| `--early-stopping-patience` | `None` (disabled) | Stop if `eval_loss` doesn't improve for this many consecutive evals. |
| `--seed` | `42` | RNG seed. |
| `--wandb-project` | `orena-frame-sft` | W&B project name. |
| `--run-name` | auto (`{model}-{lora\|full}-{timestamp}`) | W&B run name; also used to name `--output-dir` if that's unset. |

## `sft_train_qwen_frame.slurm`

Wraps the training script for submission to the `gpu-large` partition on
`gpu38` (2× H200 ~144GB, 4× RTX PRO 6000 ~98GB — SSH to the node is
disabled, all access is via Slurm). Defaults to 1× `rtx_pro_6000` (LoRA on a
9B model fits comfortably; H200s are scarcer and reserved for jobs that
actually need >98GB). All paths are absolute, anchored at
`REPO_ROOT=/home/ajenane/orena`, since `#SBATCH` directives and `sbatch`'s
working directory depend on where you invoke it from, not where the script
file lives. `TRAIN_FILE`/`EVAL_FILE`/`OUTPUT_DIR`/`EXTRA_ARGS` are all
overridable via `sbatch --export=ALL,VAR=value`; `OUTPUT_DIR` is left unset
by default so the Python script's own run-name-based default takes over.

## Known issues encountered and fixed along the way
- **`ValueError: mm_token_type_ids is missing`** — the custom collator
  originally dropped this processor output; fixed by capturing and padding
  it (see design choices above).
- **HF Hub 504 Gateway Timeout on `lapchole`** — transient outage on
  Hugging Face's side, resolved on its own; `lapchole` now loads
  successfully (see [heico_data_structure.md](heico_data_structure.md) for
  the investigation).
- **Path breakage after moving the pipeline into `orena_sft/`** — the SLURM
  script's `#SBATCH --output`/`--error` directives can't use shell
  variables (parsed before the script body runs), so those needed hardcoded
  absolute paths; the Python scripts were made location-independent via
  `Path(__file__).resolve().parent` instead of relying on cwd.
