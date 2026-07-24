#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0-06:00:00
#SBATCH --job-name=gemma-frame-eval
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Evaluates a trained gemma-4-31B checkpoint on the FOCUS frame track via
# evaluate_gemma_frame.py. CHECKPOINT_DIR is required. Defaults to an h200
# (31B ~62GB + GPU judge ~16GB needs the large card).
#
#   sbatch --export=ALL,CHECKPOINT_DIR=/home/ajenane/orena/orena_sft/checkpoints/gemma-4-31b-heico-only,EXTRA_ARGS="--datasets lapchole" \
#     orena_sft/evaluate_gemma_frame.slurm
#
# Quick sanity check:
#   sbatch --export=ALL,CHECKPOINT_DIR=...,EXTRA_ARGS="--limit 20" \
#     orena_sft/evaluate_gemma_frame.slurm

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
    echo "CHECKPOINT_DIR is required, e.g.:" >&2
    echo "  sbatch --export=ALL,CHECKPOINT_DIR=... orena_sft/evaluate_gemma_frame.slurm" >&2
    exit 1
fi

EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/evaluate_gemma_frame.py" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    $EXTRA_ARGS
