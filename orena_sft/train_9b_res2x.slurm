#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen-9b-res2x
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# 2x-resolution A/B against combined-9b-8r-direct (orena_count Stage 4: the
# base model gains number 0.092 -> 0.234 at 2x, but a native-res adapter breaks
# on 2x input -- the gain is only reachable by TRAINING at 2x). Hyperparameters
# are the original run's (effective batch 32, lr 1e-4, 3 epochs, eval/save 50,
# seed 42, LoRA r8/a16 LM-only); the ONLY changes are --min-pixels 2073600 and
# the batch layout: 16 per device x 2 H200s (DDP) = effective 32, no grad
# accumulation. Visual tokens go ~510 -> ~2040 per frame, so 32 on one GPU
# cannot fit.
#
# Preflight gates on OOM at the real 2x token load (per-device batch). Batch 16
# OOMs on a 141GB H200 (measured, job 13472), hence 8 x grad-accum 2 per device.
#
# EVALUATION: this checkpoint MUST be evaluated at the same resolution:
#   sbatch --export=ALL,MODEL=orena_sft/checkpoints/combined-9b-8r-direct-res2x \
#       orena_count/eval_res.slurm
# (eval_res.slurm defaults MINPX=2073600; evaluating without it re-creates the
# Stage 4 OOD collapse in reverse.)

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MIN_PIXELS=2073600
export SMOKE_BATCH=8
export SMOKE_WORST_CASE=1

echo "=== PREFLIGHT: memory at 2x resolution (per-device batch $SMOKE_BATCH) ==="
"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/preflight_vision_lora.py"
echo "=== PREFLIGHT PASSED -- starting training ==="

"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=2 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$SFT_DIR/sft_export/combined/train.jsonl" \
    --eval-file "$SFT_DIR/sft_export/combined/eval.jsonl" \
    --run-name combined-9b-8r-direct-res2x \
    --prompt-style direct \
    --min-pixels 2073600 \
    --lora-r 8 --lora-alpha 16 \
    --epochs 3 --lr 1e-4 --seed 42 \
    --batch-size 8 --grad-accum 2 \
    --eval-steps 50 --save-steps 50 \
    --early-stopping-patience 3
