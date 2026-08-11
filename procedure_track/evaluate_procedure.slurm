#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen-procedure-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Evaluation for the procedure track. One 96 GB RTX PRO 6000 is the same class of GPU
# the challenge runs on, so the latency numbers here are the ones a submission sees.
#
# 3,127 test prefixes at ~15 s each is ~13 h on one GPU -- shard it:
#   for i in 0 1 2 3; do
#     sbatch --export=ALL,SHARD=$i,NUM_SHARDS=4,MODE=generate,CHECKPOINT=/path/to/run \
#       procedure_track/evaluate_procedure.slurm
#   done
#   venv3.12/bin/python procedure_track/evaluate_procedure.py --mode score \
#       --checkpoint-dir /path/to/run
#
# Baseline arm (no CHECKPOINT) scores the untuned base model, which is what makes the
# SFT number interpretable -- the segment-track finding was that base VLMs are
# non-compliant rather than blind.
#
# venv3.12, not .venv: flash-linear-attention matters at prefill exactly as it does in
# training, and the judge goes through the same kernels.

set -euo pipefail

# Submit from the repo root; $PWD keeps this working on either cluster.
REPO_ROOT="${REPO_ROOT:-$PWD}"
PROC_DIR="$REPO_ROOT/procedure_track"
mkdir -p "$REPO_ROOT/logs" "$PROC_DIR/logs"

VENV="${VENV:-$([ -x "$REPO_ROOT/.venv/bin/python" ] && echo "$REPO_ROOT/.venv" || echo "$REPO_ROOT/venv3.12")}"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MODE="${MODE:-generate}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CHECKPOINT="${CHECKPOINT:-}"
CKPT_ARGS=()
[[ -n "$CHECKPOINT" ]] && CKPT_ARGS=(--checkpoint-dir "$CHECKPOINT")

echo "Node: $(hostname) | shard $SHARD/$NUM_SHARDS | mode $MODE | venv $VENV"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

export TOKENIZERS_PARALLELISM=false

"$VENV/bin/python" "$PROC_DIR/evaluate_procedure.py" \
    --mode "$MODE" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --resume \
    "${CKPT_ARGS[@]}" \
    $EXTRA_ARGS
