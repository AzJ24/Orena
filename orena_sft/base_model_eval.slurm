#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=0-04:00:00
#SBATCH --job-name=qwen-base-eval
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Evaluates a BASE (un-fine-tuned) model on the FOCUS frame track via
# base_model_eval.py -- the "is the SFT forgetting?" baseline.
#
# MODEL_ID and EXTRA_ARGS are overridable at submit time. For a 27B base you
# MUST also override --gres to an h200 (27B ~56GB weights + GPU judge ~16GB
# won't fit safely on a 98GB rtx_pro_6000):
#
#   sbatch --gres=gpu:h200:1 \
#     --export=ALL,MODEL_ID=Qwen/Qwen3.5-27B,EXTRA_ARGS="--datasets lapchole" \
#     orena_sft/base_model_eval.slurm
#
# The default gres (rtx_pro_6000) is fine for a 9B base.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/base_model_eval.py" \
    --model-id "$MODEL_ID" \
    $EXTRA_ARGS
