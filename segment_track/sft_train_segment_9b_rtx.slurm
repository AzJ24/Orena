#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=qwen9b-segment-sft
#SBATCH --output=/home/ajenane/orena/segment_track/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/segment_track/logs/%x-%j.err

# Qwen3.5-9B arm of the segment-track comparison, 2x RTX PRO 6000, on the same
# node as the 27B run. Calls sft_train_qwen_segment_ddp.py unchanged -- every
# difference is a flag, so the two arms share one code path.
#
# What is held identical to the 27B run, and must stay that way for the
# comparison to mean anything:
#   * train/eval JSONL (same export, same N=80 frames, same video-level split)
#   * effective batch 32 (1 x 16 x 2 GPUs), lr 1e-4, 3 epochs, seed 42
#   * LoRA r=8 alpha=16 on the same 7 projection names -- which land on all MLPs
#     plus the full-attention layers in both models (128 modules vs 256)
#   * direct prompt, eval every 50 steps on the same seed-42 stratified subset,
#     early stopping patience 3
#
# What differs, and why it does not affect the accuracy comparison:
#   * GPU class. Step time is NOT comparable across H200 and RTX PRO 6000; only
#     the eval_loss and accuracy curves are.
#   * 24 CPUs / 10 dataloader workers per rank, not 32/16 -- the 27B job holds 32
#     of this node's 64 CPUs. Throughput only; the data seen is identical.
#
# 9B in bf16 is ~17 GB of weights against 96 GB, so gradient checkpointing is not
# forced here as it is at 27B. It is left ON regardless: turning it off would
# change the arithmetic the 27B arm ran under for no gain in a comparison.
#
#   sbatch --export=ALL,EXTRA_ARGS="--max-steps 20 --run-name smoke-9b" \
#       segment_track/sft_train_segment_9b_rtx.slurm
#   sbatch segment_track/sft_train_segment_9b_rtx.slurm

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SEG_DIR="$REPO_ROOT/segment_track"
mkdir -p "$SEG_DIR/logs"

VENV="${VENV:-$REPO_ROOT/venv3.12}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
TRAIN_FILE="${TRAIN_FILE:-$SEG_DIR/sft_export/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-$SEG_DIR/sft_export/eval.jsonl}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Node: $(hostname) | model: $MODEL_ID"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

OUTPUT_DIR_ARGS=()
if [[ -n "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR_ARGS=(--output-dir "$OUTPUT_DIR")
fi

export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

echo "venv: $VENV"
"$VENV/bin/torchrun" --standalone --nproc_per_node=2 \
    "$SEG_DIR/sft_train_qwen_segment_ddp.py" \
    --model-id "$MODEL_ID" \
    --train-file "$TRAIN_FILE" \
    --eval-file "$EVAL_FILE" \
    --dataloader-workers "$DATALOADER_WORKERS" \
    "${OUTPUT_DIR_ARGS[@]}" \
    $EXTRA_ARGS
