#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=0-23:00:00
#SBATCH --job-name=qwen-frame-sft-ddp
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# DDP training for Qwen3.5-9B on the FOCUS frame track, across 2x H200 on one
# node via torchrun (one process per GPU). Data-parallel: each GPU holds a full
# 9B+LoRA replica and processes a data shard; gradients averaged every step.
#
# Keep the EFFECTIVE batch equal to the single-GPU runs (32) so convergence
# matches: with 2 GPUs use --batch-size 16 (16 x 2 = 32). See orena_sft/ddp.md.
#
#   sbatch --export=ALL,TRAIN_FILE=...,EVAL_FILE=...,EXTRA_ARGS="--model-id \
#     Qwen/Qwen3.5-9B --prompt-style direct --run-name combined-all-9b-8r-direct-ddp \
#     --lora-r 8 --lora-alpha 16 --epochs 1.5 --batch-size 16 --grad-accum 1 \
#     --save-steps 150 --eval-steps 150" orena_sft/sft_train_qwen_frame_ddp.slurm

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

# NCCL over PCIe/NVLink on a single node; disable IB probing to avoid stalls.
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

# one process per GPU, single node. NPROC=1 reproduces the single-GPU runs
# (pass --batch-size 32 --grad-accum 1 to keep the effective batch at 32);
# override the matching --gres on the sbatch command line.
"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node="${NPROC:-2}" \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$TRAIN_FILE" \
    --eval-file "$EVAL_FILE" \
    "${OUTPUT_DIR_ARGS[@]}" \
    $EXTRA_ARGS
