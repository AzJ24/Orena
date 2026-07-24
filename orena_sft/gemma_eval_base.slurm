#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0-06:00:00
#SBATCH --job-name=gemma-base-eval
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Evaluates BASE google/gemma-4-31B-it (no fine-tuning) on the FOCUS frame
# track via gemma_eval_base.py -- the "is the SFT forgetting?" baseline.
# Defaults to an h200 (31B ~62GB + GPU judge ~16GB needs the large card).
#
#   sbatch --export=ALL,EXTRA_ARGS="--datasets lapchole" \
#     orena_sft/gemma_eval_base.slurm
#
# MODEL_ID overridable (e.g. a smaller Gemma for a quick check).

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

MODEL_ID="${MODEL_ID:-google/gemma-4-31B-it}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/gemma_eval_base.py" \
    --model-id "$MODEL_ID" \
    $EXTRA_ARGS
