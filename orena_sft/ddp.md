# Multi-GPU (DDP) training for the FOCUS SFT

How to run `sft_train_qwen_frame.py` across multiple GPUs (e.g. 8× H100) with
**DistributedDataParallel (DDP)**, what must change, and what to expect vs the
single-GPU run.

---

## 1. Why DDP (data-parallel), not model-parallel

A 9B model + LoRA fits comfortably on **one** 80 GB H100 (~18 GB weights in bf16 +
activations). So we don't need to *split the model* — we want **8 full replicas,
each processing a different data shard**, gradients averaged every step → ~8×
throughput. That is DDP.

**The current script does NOT do this.** It loads the model with
`device_map="auto"` ([sft_train_qwen_frame.py](sft_train_qwen_frame.py) model-load
block), which does **naive model-parallel** (one model sharded across GPUs in one
process). That is *incompatible* with DDP and must be changed.

The trainer itself (TRL `SFTTrainer`, built on HF `Trainer`) **supports DDP
natively** — the only reason it doesn't run distributed today is `device_map="auto"`
plus a plain `python` launcher.

---

## 2. Will DDP give the same results as single-GPU?

**Not bit-identical — ever.** DDP all-reduces gradients across GPUs in a different
floating-point order (FP addition isn't associative), shards data differently
(`DistributedSampler`), and draws dropout RNG per rank. The final weights differ.

The question is *equivalent quality* vs *different training*:

| config | effective batch | vs single-GPU (batch 32) |
|---|---|---|
| **per-device batch = 4** on 8 GPUs | 4 × 8 = **32** | **equivalent quality** (like a different seed), not identical |
| per-device batch = 32 on 8 GPUs | 32 × 8 = **256** | **different** training — 8× larger batch, needs LR re-tuning |

**To preserve the single-GPU behaviour (and the ~1.5-epoch convergence estimate),
keep the effective batch at 32 → set `--batch-size 4`.** If you instead let the
effective batch grow to 256, treat it as a new hyperparameter regime (scale LR up,
re-tune epochs); results will differ, often slightly worse without tuning.

---

## 3. Required changes

### 3a. Model load — place each replica on its local rank
Replace `device_map="auto"` so each DDP process loads a **full** model on its own
GPU (keep an `"auto"` fallback for single-GPU runs):

```python
import os
...
local_rank = int(os.environ.get("LOCAL_RANK", -1))
device_map = {"": local_rank} if local_rank >= 0 else "auto"
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    args.model_id, dtype=torch.bfloat16, device_map=device_map,
)
```

### 3b. Rank-guard the generation callback (prevents deadlock)
`SampleGenerationCallback.on_evaluate` calls `model.generate()`. Under DDP that runs
on all 8 ranks and can **hang** (generation is a forward-only pass DDP doesn't
expect). Guard it to the main process and unwrap the DDP wrapper:

```python
def on_evaluate(self, args, state, control, model=None, **kwargs):
    if not state.is_world_process_zero:
        return
    model = getattr(model, "module", model)   # unwrap DDP
    ...
```
(Apply the same `is_world_process_zero` guard to any other rank-0-only work —
`preview_example()` printing, wandb sample logging.)

### 3c. PEFT + DDP unused-parameters
With LoRA the base + vision tower are frozen, so DDP may raise "parameters didn't
receive grad." Add to `SFTConfig`:
```python
ddp_find_unused_parameters=False,   # if it errors, set True
```

### 3d. Effective batch / LR
Set `--batch-size 4` (per-device) to keep effective batch = 32 (see §2). If you
deliberately want the big batch, scale `--lr` up and re-tune `--epochs`.

---

## 4. The launcher — `torchrun`, not `python`

DDP needs one process **per GPU**, spawned by `torchrun`. A DDP slurm
(`sft_train_qwen_frame_ddp.slurm`) differs from the single-GPU one only in the
resource request and the launch line:

```bash
#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=0-23:00:00
#SBATCH --job-name=qwen-frame-sft-ddp
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

set -euo pipefail
REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

TRAIN_FILE="${TRAIN_FILE:-$SFT_DIR/sft_export/combined/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-$SFT_DIR/sft_export/combined/eval.jsonl}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv

# one process per GPU on this node
"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=8 \
    "$SFT_DIR/sft_train_qwen_frame.py" \
    --train-file "$TRAIN_FILE" \
    --eval-file "$EVAL_FILE" \
    $EXTRA_ARGS
```

Notes:
- `--standalone --nproc_per_node=8` = single-node, 8 processes (one per GPU).
- `torchrun` sets `LOCAL_RANK`/`RANK`/`WORLD_SIZE`, which §3a reads.
- `--cpus-per-task`/`--mem` bumped for 8 data-loader groups.
- Multi-node would use `--nnodes` + a rendezvous endpoint (not needed for 8×1-node).

---

## 5. How to run

```bash
sbatch --export=ALL,\
TRAIN_FILE=/home/ajenane/orena/orena_sft/sft_export/combined_all/train.jsonl,\
EVAL_FILE=/home/ajenane/orena/orena_sft/sft_export/combined/eval.jsonl,\
EXTRA_ARGS="--model-id Qwen/Qwen3.5-9B --prompt-style direct \
  --run-name combined-all-9b-8r-direct-ddp --lora-r 8 --lora-alpha 16 \
  --epochs 1.5 --batch-size 4 --grad-accum 1 --save-steps 150 --eval-steps 150" \
  orena_sft/sft_train_qwen_frame_ddp.slurm
```

- **`--batch-size 4`** → effective batch 4×8 = 32 (matches the single-GPU runs, so
  the ~1.5-epoch convergence estimate holds).
- Steps/epoch = `n_examples / (batch × 8)` = e.g. 20000 / 32 ≈ 625, same as
  single-GPU batch-32 — so `--save-steps`/`--eval-steps` keep the same meaning.
- Checkpoints, wandb, and `save_total_limit` behave as before (Trainer writes only
  on rank 0).

---

## 6. Checklist

- [ ] `device_map` reads `LOCAL_RANK` (§3a)
- [ ] generation callback guarded to `is_world_process_zero` + DDP-unwrapped (§3b)
- [ ] `ddp_find_unused_parameters` set in `SFTConfig` (§3c)
- [ ] `--batch-size 4` to keep effective batch 32 (§3d) — or re-tune LR for big batch
- [ ] launched via `torchrun --nproc_per_node=8` on `gpu:h100:8` (§4)
- [ ] single-GPU `sft_train_qwen_frame.slurm` left untouched (both paths work)

---

## 7. Expectation

~8× wall-clock speedup (minus all-reduce overhead). With effective batch matched to
32, the model is **quality-equivalent** to the single-GPU run (a different seed's
trajectory), **not** a bit-identical reproduction. If you need identical weights,
only the single-GPU config with a fixed seed reproduces itself.
