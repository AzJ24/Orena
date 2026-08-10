#!/bin/bash
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen27b-alldata
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# FINAL SUBMISSION RUN -- Qwen3.6-27B on ALL 20000 labelled rows (train + eval
# + the public test split), 8x H100 DDP. The platform scores against its own
# hidden set, so the labelled public test split is just more training data here.
#
# CONSEQUENCE: this checkpoint cannot be evaluated locally. Every row of our
# test set is in its training data, so evaluate_qwen_frame.py would report
# memorisation. Judge it only on the submission platform.
#
# There is NO held-out eval, so the step count is fixed in advance from the
# v2-split runs: 27B's best checkpoint landed at epoch 1.20 (9B's at 1.44-1.51).
# 20000 rows / effective batch 32 = 625 steps per epoch, so 1.2 epochs = 750
# steps. --epochs 1.2 expresses that directly.
#
# Effective batch must stay 32 to match every run the epoch estimate came from:
# 8 GPUs x per-device 4 x grad_accum 1 = 32.
#
# MEMORY: 27B needed 92.5 GiB per device at batch 8 on a 141 GB H200. An 80 GB
# H100 cannot hold that, hence per-device 4. The preflight measures it for real
# before the run starts -- if batch 4 OOMs, use 2 with --grad-accum 2 and
# --nproc_per_node 8 (still 32), or fall back to 16 GPUs.
#
# Cluster: partition 'all' carries gpu:h100:8 nodes (cn01-35); an 'h200'
# partition with gpu:h200:8 also exists. H100 is the right target -- 8 GPUs at
# the required effective batch 32 means per-device 4 either way, so the H200's
# extra memory buys headroom, not throughput. Swap to --partition=h200
# --gres=gpu:h200:8 only if the preflight cannot fit batch 4 on 80 GB.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
SFT_DIR="$REPO_ROOT/orena_sft"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/vast/workspaces/VL_LeJepa/data/orena}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
mkdir -p "$REPO_ROOT/logs"

echo "Node: $(hostname) | repo=$REPO_ROOT | data=$DATA_ROOT | model=$MODEL_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

for d in heico lapchole; do
    [[ -d "$DATA_ROOT/$d/frames" ]] || { echo "MISSING: $DATA_ROOT/$d/frames" >&2; exit 1; }
done

export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# wandb.init() raises on an unconfigured cluster and takes the whole run with
# it. Log offline instead -- the run is still recorded and can be uploaded later
# with `wandb sync wandb/offline-run-*`. Checking credentials rather than
# calling the API keeps this from hanging on a compute node with no network.
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    export WANDB_MODE=offline
    echo "wandb: no credentials found -> WANDB_MODE=offline"
fi

# Weights must already be cached: 8 ranks resolving against the Hub race each
# other, and compute nodes are usually offline. Run prefetch_assets.sh on a
# login node first.
if "$PY" -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_files_only=True)" 2>/dev/null; then
    export HF_HUB_OFFLINE=1
    echo "model cached -> HF_HUB_OFFLINE=1"
else
    echo "ERROR: $MODEL_ID is not in the HF cache." >&2
    echo "Run on a LOGIN node first:  bash orena_sft/prefetch_assets.sh" >&2
    exit 1
fi

# The export is gitignored and stores ABSOLUTE frame paths, so it must be
# rebuilt here rather than copied from the other cluster.
if [[ ! -f "$SFT_DIR/sft_export/combined_alldata_test/train.jsonl" ]]; then
    echo "=== building all-data export (train + eval + public test = 20000 rows) ==="
    "$PY" "$SFT_DIR/build_frame_sft_dataset.py" \
        --datasets heico lapchole --all-data --include-test --root-dir "$DATA_ROOT" \
        --out-dir "$SFT_DIR/sft_export/combined_alldata_test" \
        --manifest "$SFT_DIR/split_manifest_alldata_test.json"
fi

# All-data mode has no held-out set, but the Trainer still wants an eval_dataset.
# Use a small slice of train purely as a divergence check -- its loss is a
# training-loss proxy and MUST NOT be used to pick a checkpoint.
MONITOR="$SFT_DIR/sft_export/combined_alldata_test/monitor.jsonl"
[[ -f "$MONITOR" ]] || head -n 400 "$SFT_DIR/sft_export/combined_alldata_test/train.jsonl" > "$MONITOR"

# LORA_VISION=1 also adapts the vision tower + patch merger. On the 9B this was
# the only intervention whose gain replicated significantly on both datasets
# (object_recognition +0.023 / +0.025), but it costs activation memory -- it
# forced batch 32 -> 16 there, so expect the sweep to land lower here too.
VISION_ARGS=()
if [[ "${LORA_VISION:-0}" == "1" ]]; then
    VISION_ARGS=(--lora-vision)
    RUN_NAME="combined-27b-8r-direct-vision-alldata-withtest"
    export LORA_VISION=1
    export SMOKE_BATCH="${SMOKE_BATCH:-4,2,1}"
else
    RUN_NAME="combined-27b-8r-direct-alldata-withtest"
    export SMOKE_BATCH="${SMOKE_BATCH:-4,2}"
fi

export SMOKE_MODEL_ID="$MODEL_ID"
export SMOKE_WORST_CASE=1
export SMOKE_TRAIN_FILE="$SFT_DIR/sft_export/combined_alldata_test/train.jsonl"
echo "=== PREFLIGHT: 27B${VISION_ARGS:+ +vision} memory (sweeps $SMOKE_BATCH) ==="
"$PY" "$SFT_DIR/preflight_vision_lora.py"
echo "=== PREFLIGHT PASSED -- run=$RUN_NAME ==="

# Effective batch must stay 32. With 8 GPUs: per-device 4 -> accum 1,
# per-device 2 -> accum 2, per-device 1 -> accum 4.
PER_DEV="${PER_DEV:-4}"
ACCUM=$(( 32 / (8 * PER_DEV) ))
echo "batch layout: 8 GPUs x $PER_DEV x accum $ACCUM = $(( 8 * PER_DEV * ACCUM ))"

"${PY%/*}/torchrun" --standalone --nproc_per_node=8 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$SFT_DIR/sft_export/combined_alldata_test/train.jsonl" \
    --eval-file  "$MONITOR" \
    --model-id "$MODEL_ID" \
    --run-name "$RUN_NAME" \
    --prompt-style direct \
    "${VISION_ARGS[@]}" \
    --lora-r 8 --lora-alpha 16 \
    --epochs 1.2 --lr 1e-4 --seed 42 \
    --batch-size "$PER_DEV" --grad-accum "$ACCUM" \
    --eval-steps 100 --save-steps 100
