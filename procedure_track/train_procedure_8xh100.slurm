#!/bin/bash
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=0-08:00:00
#SBATCH --job-name=qwen-procedure-8xh100
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Qwen3.5-9B LoRA SFT on the FOCUS procedure track, 8x H100 DDP.
# Cluster conventions match orena_sft/train_27b_alldata_8gpu.slurm: partition 'all'
# carries the gpu:h100:8 nodes, and the frames live under DATA_ROOT on /mnt/vast.
#
# MEMORY -- read this before changing --num-frames or the batch layout.
# A procedure prompt is ~66,000 tokens and ~10 of them are training targets. The stock
# causal-LM path runs lm_head over the whole sequence and upcasts to fp32:
#     66,300 x 248,320 vocab x 2 B = 32.9 GB bf16, x4 B = 65.9 GB fp32  -> ~99 GB
# which cannot fit an 80 GB H100 at any batch size. sparse_logits_loss.py computes
# logits only at supervised positions (~15 MB instead of 33 GB); the loss and gradients
# are asserted bit-equal to the full-logits reference in tests/test_sparse_logits.py.
# Without it this job OOMs in the first forward pass.
#
# BATCH LAYOUT. Effective batch stays 32 to match the frame and segment runs, so their
# epoch estimates carry over: 8 GPUs x PER_DEV 1 x accum 4. Per-device must be 1 -- a
# prefix is up to 768 frames, ~50x a frame-track sample.
#
# DATA. The export stores absolute frame_dir paths from wherever it was built, so
# --frames-root re-roots them onto this cluster. Copy the JSONL over (train 11 MB,
# eval 1.3 MB) rather than re-exporting:
#     rsync -av procedure_track/sft_export/ cluster43:$REPO_ROOT/procedure_track/sft_export/
#
#   sbatch --export=ALL,EXTRA_ARGS="--max-steps 20 --run-name smoke" \
#       procedure_track/train_procedure_8xh100.slurm
#   sbatch procedure_track/train_procedure_8xh100.slurm

set -euo pipefail

# Submit from the repo root. Deriving it from $PWD is what makes one file work on
# both clusters -- home is /home/ajenane here and /mnt/vast/home/aj45nyja there.
REPO_ROOT="${REPO_ROOT:-$PWD}"
PROC_DIR="$REPO_ROOT/procedure_track"
DATA_ROOT="${DATA_ROOT:-/mnt/vast/workspaces/VL_LeJepa/data/orena}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
TRAIN_FILE="${TRAIN_FILE:-$PROC_DIR/sft_export/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-$PROC_DIR/sft_export/eval.jsonl}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
mkdir -p "$REPO_ROOT/logs" "$PROC_DIR/logs"

# fla supplies the fused Gated DeltaNet kernels for the linear-attention layers; without
# it transformers falls back to a Python loop, measured 4x slower on the segment track.
# Pick whichever venv actually has it rather than hardcoding one.
# This cluster ships .venv; the build cluster uses venv3.12. Prefer whichever has fla,
# fall back to the first that merely exists.
PY="${PY:-}"
if [[ -z "$PY" ]]; then
    for cand in "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/venv3.12/bin/python"; do
        [[ -x "$cand" ]] || continue
        if "$cand" -c "import fla" 2>/dev/null; then
            PY="$cand"
            break
        fi
        if [[ -z "$PY" ]]; then
            PY="$cand"
        fi
    done
fi
[[ -x "${PY:-}" ]] || { echo "no usable python found under $REPO_ROOT; set PY=" >&2; exit 1; }

echo "Node: $(hostname)"
echo "repo=$REPO_ROOT  data=$DATA_ROOT  model=$MODEL_ID"
echo "python=$PY  fla=$("$PY" -c 'import importlib.util as u; print(u.find_spec("fla") is not None)')"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Fail here rather than 20 minutes in. The export points at frames_overlay -- the
# burned-in clock is what the prompt tells the model to read.
for d in heico lapchole; do
    [[ -d "$DATA_ROOT/$d/frames_overlay" ]] \
        || { echo "MISSING: $DATA_ROOT/$d/frames_overlay" >&2; exit 1; }
done
for f in "$TRAIN_FILE" "$EVAL_FILE"; do
    [[ -s "$f" ]] || { echo "MISSING export: $f  (rsync it from the build cluster)" >&2; exit 1; }
done

export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 8 ranks x dataloader workers on one node oversubscribes the CPUs without this.
export OMP_NUM_THREADS=4
# Fail loudly on an NCCL desync instead of hanging until the wall clock expires.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# wandb.init() raises on an unconfigured cluster and takes the whole run with it. Log
# offline instead; upload later with `wandb sync wandb/offline-run-*`. Checking
# credentials rather than calling the API keeps this from hanging on an offline node.
# The H100 cluster keeps WANDB_API_KEY in a repo-local .env; `set -a` exports it.
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    export WANDB_MODE=offline
    echo "wandb: no credentials found -> WANDB_MODE=offline"
fi

# 8 ranks resolving against the Hub race each other, and compute nodes are usually
# offline. Prefetch on a login node first.
if "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_files_only=True)" 2>/dev/null; then
    export HF_HUB_OFFLINE=1
    echo "model cached -> HF_HUB_OFFLINE=1"
else
    echo "ERROR: $MODEL_ID is not in the HF cache." >&2
    echo "On a LOGIN node:  $PY -c \"from huggingface_hub import snapshot_download as d; d('$MODEL_ID')\"" >&2
    exit 1
fi

# Effective batch must stay 32: 8 GPUs x PER_DEV x accum.
PER_DEV="${PER_DEV:-1}"
ACCUM=$(( 32 / (8 * PER_DEV) ))
(( 8 * PER_DEV * ACCUM == 32 )) || { echo "PER_DEV=$PER_DEV cannot reach 32" >&2; exit 1; }
echo "batch layout: 8 GPUs x $PER_DEV x accum $ACCUM = $(( 8 * PER_DEV * ACCUM ))"

"${PY%/*}/torchrun" --standalone --nproc_per_node=8 \
    "$PROC_DIR/sft_train_qwen_procedure_ddp.py" \
    --model-id "$MODEL_ID" \
    --train-file "$TRAIN_FILE" \
    --eval-file "$EVAL_FILE" \
    --frames-root "$DATA_ROOT" \
    --batch-size "$PER_DEV" --grad-accum "$ACCUM" \
    --lora-r 8 --lora-alpha 16 \
    --lr 1e-4 --seed 42 \
    --eval-steps 50 --save-steps 50 \
    --dataloader-workers 4 \
    $EXTRA_ARGS
