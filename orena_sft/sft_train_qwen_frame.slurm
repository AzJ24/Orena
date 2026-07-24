#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=0-23:00:00
#SBATCH --job-name=qwen-frame-sft
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Paths are absolute (anchored at REPO_ROOT below) since #SBATCH directives
# and sbatch's working directory depend on where you invoke `sbatch` from,
# not where this script lives.
#
# Override paths/hyperparameters at submission time without editing this
# file, e.g. for the smoke test:
#
#   sbatch --export=ALL,TRAIN_FILE=/home/ajenane/orena/orena_sft/smoke/train.jsonl,\
#     EVAL_FILE=/home/ajenane/orena/orena_sft/smoke/eval.jsonl,\
#     EXTRA_ARGS="--epochs 1 --batch-size 1" \
#     orena_sft/sft_train_qwen_frame.slurm
#
# OUTPUT_DIR is left unset by default -- the python script itself defaults it
# to <script-dir>/checkpoints/<run_name>, so checkpoints land in a directory
# named after the wandb run (e.g. Qwen3.5-9B-lora-2026-07-16_16-42-10).
# Only set OUTPUT_DIR here if you want to override that.
#
# For a full run, just: sbatch orena_sft/sft_train_qwen_frame.slurm

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

TRAIN_FILE="${TRAIN_FILE:-$SFT_DIR/sft_export/combined/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-$SFT_DIR/sft_export/combined/eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

OUTPUT_DIR_ARGS=()
if [[ -n "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR_ARGS=(--output-dir "$OUTPUT_DIR")
fi

"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/sft_train_qwen_frame.py" \
    --train-file "$TRAIN_FILE" \
    --eval-file "$EVAL_FILE" \
    "${OUTPUT_DIR_ARGS[@]}" \
    $EXTRA_ARGS
