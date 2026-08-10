#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen27b-alldata
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# FINAL SUBMISSION RUN -- Qwen3.6-27B on all 13748 non-test rows, 8x H100 DDP.
#
# There is NO held-out eval in all-data mode, so the step count is fixed in
# advance from the v2-split runs: 27B's best checkpoint landed at epoch 1.20
# (9B's at 1.44-1.51). 13748 rows / effective batch 32 = 430 steps per epoch,
# so 1.2 epochs = ~516 steps. --epochs 1.2 expresses that directly.
#
# Effective batch must stay 32 to match every run the epoch estimate came from:
# 8 GPUs x per-device 4 x grad_accum 1 = 32.
#
# MEMORY: 27B needed 92.5 GiB per device at batch 8 on a 141 GB H200. An 80 GB
# H100 cannot hold that, hence per-device 4. The preflight measures it for real
# before the run starts -- if batch 4 OOMs, use 2 with --grad-accum 2 and
# --nproc_per_node 8 (still 32), or fall back to 16 GPUs.
#
# Adjust partition/gres names for the target cluster's Slurm config.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
SFT_DIR="$REPO_ROOT/orena_sft"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dir containing <dataset>/frames/ on this cluster}"
mkdir -p "$REPO_ROOT/logs"

echo "Node: $(hostname) | repo=$REPO_ROOT | data=$DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # both ranks resolving against the Hub races
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The export is gitignored and stores ABSOLUTE frame paths, so it must be
# rebuilt here rather than copied from the other cluster.
if [[ ! -f "$SFT_DIR/sft_export/combined_alldata/train.jsonl" ]]; then
    echo "=== building all-data export (asserts no test leakage) ==="
    "$PY" "$SFT_DIR/build_frame_sft_dataset.py" \
        --datasets heico lapchole --all-data --root-dir "$DATA_ROOT" \
        --out-dir "$SFT_DIR/sft_export/combined_alldata" \
        --manifest "$SFT_DIR/split_manifest_alldata.json"
fi

# All-data mode has no held-out set, but the Trainer still wants an eval_dataset.
# Use a small slice of train purely as a divergence check -- its loss is a
# training-loss proxy and MUST NOT be used to pick a checkpoint.
MONITOR="$SFT_DIR/sft_export/combined_alldata/monitor.jsonl"
[[ -f "$MONITOR" ]] || head -n 400 "$SFT_DIR/sft_export/combined_alldata/train.jsonl" > "$MONITOR"

export SMOKE_MODEL_ID="Qwen/Qwen3.6-27B"
export SMOKE_BATCH="4,2"
export SMOKE_WORST_CASE=1
echo "=== PREFLIGHT: 27B memory on this GPU (sweeps $SMOKE_BATCH) ==="
"$PY" "$SFT_DIR/preflight_vision_lora.py"
echo "=== PREFLIGHT PASSED ==="

"${PY%/*}/torchrun" --standalone --nproc_per_node=8 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$SFT_DIR/sft_export/combined_alldata/train.jsonl" \
    --eval-file  "$MONITOR" \
    --model-id "Qwen/Qwen3.6-27B" \
    --run-name combined-27b-8r-direct-alldata \
    --prompt-style direct \
    --lora-r 8 --lora-alpha 16 \
    --epochs 1.2 --lr 1e-4 --seed 42 \
    --batch-size 4 --grad-accum 1 \
    --eval-steps 100 --save-steps 100
