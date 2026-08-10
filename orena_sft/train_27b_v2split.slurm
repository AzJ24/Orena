#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=qwen-27b-v2split
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# Qwen3.6-27B on the v2 eval split, single H200 (runs alongside the 9B v2 job on
# the node's other H200). Same hyperparameters as combined-9b-8r-direct and its
# v2 re-run, so 9B-vs-27B is isolated: lr 1e-4, 3 epochs, linear schedule,
# weight_decay 0, seed 42, LoRA r8/a16 LM-only, native resolution, patience 3.
#
# Batch layout differs by necessity: 27B needs ~92.5 GiB at batch 8 (measured in
# the 2x-H200 preflight; DDP ranks each hold a full replica, so per-GPU memory is
# the same single-GPU). 8 x grad_accum 4 keeps the effective batch at 32.
#
# If the preflight OOMs at 8, retry with SMOKE_BATCH=4 and
# --batch-size 4 --grad-accum 8.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== verifying v2 split against manifest ==="
"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/verify_split_manifest.py"

export SMOKE_MODEL_ID="Qwen/Qwen3.6-27B"
export SMOKE_BATCH=8
export SMOKE_TRAIN_FILE="$SFT_DIR/sft_export/combined_v2/train.jsonl"
export SMOKE_WORST_CASE=1
echo "=== PREFLIGHT: 27B memory at batch $SMOKE_BATCH (worst-case frames) ==="
"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/preflight_vision_lora.py"
echo "=== PREFLIGHT PASSED -- starting training ==="

"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=1 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$SFT_DIR/sft_export/combined_v2/train.jsonl" \
    --eval-file "$SFT_DIR/sft_export/combined_v2/eval.jsonl" \
    --model-id "Qwen/Qwen3.6-27B" \
    --run-name combined-27b-8r-direct-v2split \
    --prompt-style direct \
    --lora-r 8 --lora-alpha 16 \
    --epochs 3 --lr 1e-4 --seed 42 \
    --batch-size 8 --grad-accum 4 \
    --eval-steps 50 --save-steps 50 \
    --early-stopping-patience 3
