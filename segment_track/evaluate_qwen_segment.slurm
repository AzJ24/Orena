#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen-segment-eval
#SBATCH --output=/home/ajenane/orena/segment_track/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/segment_track/logs/%x-%j.err

# Evaluation for the segment track. One GPU is enough: inference at 64 frames is
# ~61 GB (weights 54 GB + a small KV cache, since only 16 of 64 layers use full
# attention), which fits a 96 GB RTX PRO 6000 and leaves the H200s for training.
#
# Baseline arm (no --checkpoint-dir) evaluates the untuned base model, which is
# also the Phase-0 probe: it shows whether the model reads the <n seconds> markers
# and converts them to hh:mm:ss before any training exists.
#
#   sbatch --export=ALL,EXTRA_ARGS="--limit 20" segment_track/evaluate_qwen_segment.slurm
#   sbatch --export=ALL,CHECKPOINT=/path/to/run segment_track/evaluate_qwen_segment.slurm

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SEG_DIR="$REPO_ROOT/segment_track"
mkdir -p "$SEG_DIR/logs"

CHECKPOINT="${CHECKPOINT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

CKPT_ARGS=()
if [[ -n "$CHECKPOINT" ]]; then
    CKPT_ARGS=(--checkpoint-dir "$CHECKPOINT")
fi

export TOKENIZERS_PARALLELISM=false

"$REPO_ROOT/.venv/bin/python" "$SEG_DIR/evaluate_qwen_segment.py" \
    "${CKPT_ARGS[@]}" \
    $EXTRA_ARGS
