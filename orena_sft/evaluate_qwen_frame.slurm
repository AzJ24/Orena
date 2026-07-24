#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=0-06:00:00
#SBATCH --job-name=qwen-frame-eval
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Runs orena_sft/evaluate_qwen_frame.py against a trained checkpoint.
# CHECKPOINT_DIR is required -- there's no sane default (unlike training's
# OUTPUT_DIR), since it must point at a specific finished run.
#
#   sbatch --export=ALL,CHECKPOINT_DIR=/home/ajenane/orena/orena_sft/checkpoints/Qwen3.5-9B-lora-2026-07-16_16-41-57 \
#     orena_sft/evaluate_qwen_frame.slurm
#
# Quick sanity check on a handful of examples first:
#
#   sbatch --export=ALL,CHECKPOINT_DIR=...,EXTRA_ARGS="--limit 20" \
#     orena_sft/evaluate_qwen_frame.slurm
#
# 6h time limit is generous headroom for sequential (non-batched) generation
# over the full test set (up to 6,252 examples across both datasets) plus
# LLM-judge scoring on CPU; bump with --time if you evaluate more datasets
# or raise --max-new-tokens.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
    echo "CHECKPOINT_DIR is required, e.g.:" >&2
    echo "  sbatch --export=ALL,CHECKPOINT_DIR=... orena_sft/evaluate_qwen_frame.slurm" >&2
    exit 1
fi

EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/evaluate_qwen_frame.py" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    $EXTRA_ARGS
