#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen-segment-base
#SBATCH --output=/home/ajenane/orena/segment_track/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/segment_track/logs/%x-%j.err

# Baseline: untuned Qwen3.6-27B over the SEGMENT test split, one RTX PRO 6000.
# Inference is ~61 GB (54 GB weights + a small KV cache -- only 16 of 64 layers
# use full attention) plus the ~9 GB Qwen3.5-4B judge, inside 96 GB.
#
# venv3.12, not .venv: flash-linear-attention matters at prefill exactly as it
# does in training (48 linear-attention layers over a ~9.9k-token clip), and
# orena-focus 0.3.5 is the version the checkpoints will be scored with.
# 8 CPUs and 96 GB leave the concurrent training job its 32 CPUs on this node.
#
#   for i in 0 1 2 3; do
#     sbatch --export=ALL,SHARD=$i,NUM_SHARDS=4 segment_track/evaluate_base_segment.slurm
#   done
#   venv3.12/bin/python segment_track/evaluate_base_segment.py --mode score

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SEG_DIR="$REPO_ROOT/segment_track"
mkdir -p "$SEG_DIR/logs"

VENV="${VENV:-$REPO_ROOT/venv3.12}"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MODE="${MODE:-generate}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CHECKPOINT="${CHECKPOINT:-}"
CKPT_ARGS=()
[[ -n "$CHECKPOINT" ]] && CKPT_ARGS=(--checkpoint-dir "$CHECKPOINT")

echo "Node: $(hostname) | shard $SHARD/$NUM_SHARDS | venv $VENV"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

export TOKENIZERS_PARALLELISM=false

"$VENV/bin/python" "$SEG_DIR/evaluate_base_segment.py" \
    --mode "$MODE" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS" \
    --resume \
    "${CKPT_ARGS[@]}" \
    $EXTRA_ARGS
