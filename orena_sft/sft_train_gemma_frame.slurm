#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0-23:00:00
#SBATCH --job-name=gemma-frame-sft
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# SFT of google/gemma-4-31B-it on the FOCUS frame track
# (orena_sft/sft_train_gemma_frame.py).
#
# Defaults to an h200 (144GB): gemma-4-31B is ~62GB bf16 weights, which needs
# the large card for LoRA training with activations -- an rtx_pro_6000 (98GB)
# would be tight/OOM. Bump --mem / lower --batch-size if needed.
#
# Smoke test first (recommended, catches OOM in ~10min before a multi-hour run):
#   sbatch --export=ALL,TRAIN_FILE=/home/ajenane/orena/orena_sft/sft_export/heico/train.jsonl,\
#     EVAL_FILE=/home/ajenane/orena/orena_sft/sft_export/heico/eval.jsonl,\
#     EXTRA_ARGS="--batch-size 16 --grad-accum 2 --max-steps 10 --eval-steps 5 --save-steps 1000 --run-name gemma-4-31b-smoke" \
#     orena_sft/sft_train_gemma_frame.slurm
#
# Full run:
#   sbatch --export=ALL,TRAIN_FILE=...heico/train.jsonl,EVAL_FILE=...heico/eval.jsonl,\
#     EXTRA_ARGS="--epochs 3 --batch-size 16 --grad-accum 2 --eval-steps 50 --save-steps 50 --early-stopping-patience 3 --run-name gemma-4-31b-heico-only" \
#     orena_sft/sft_train_gemma_frame.slurm
#
# OUTPUT_DIR is left unset -> the script names it checkpoints/<run_name>.

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

"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/sft_train_gemma_frame.py" \
    --train-file "$TRAIN_FILE" \
    --eval-file "$EVAL_FILE" \
    "${OUTPUT_DIR_ARGS[@]}" \
    $EXTRA_ARGS
