#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen-9b-v2split
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# combined-9b-8r-direct re-run on the v2 eval split. Hyperparameters are the
# originals recovered from that run's training_args.bin (single GPU, per-device
# batch 32, grad_accum 1, lr 1e-4, 3 epochs, eval/save every 50, linear
# schedule, weight_decay 0, seed 42, LoRA r8/a16 LM-only, native resolution);
# the ONLY change is --train-file/--eval-file pointing at combined_v2.
#
# Single GPU on purpose: the original was single-GPU, and DDP would reshard the
# data across ranks, changing which examples land in each optimizer step.
#
# v2 holds out more videos, so train is 10625 rows (was 12225) -> ~332 steps per
# epoch, ~996 for 3 epochs (was ~1146). Eval-loss values are NOT comparable to
# the original run's 0.3257: v2 is a different eval set. The comparison that
# matters is test accuracy from evaluate_qwen_frame.py.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail fast if the export drifted from the pinned manifest: a silently different
# split would make this run incomparable to everything built on top of it.
echo "=== verifying v2 split against manifest ==="
"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/verify_split_manifest.py"

"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=1 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$SFT_DIR/sft_export/combined_v2/train.jsonl" \
    --eval-file "$SFT_DIR/sft_export/combined_v2/eval.jsonl" \
    --run-name combined-9b-8r-direct-v2split \
    --prompt-style direct \
    --lora-r 8 --lora-alpha 16 \
    --epochs 3 --lr 1e-4 --seed 42 \
    --batch-size 32 --grad-accum 1 \
    --eval-steps 50 --save-steps 50 \
    --early-stopping-patience 3
