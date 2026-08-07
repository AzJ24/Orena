#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=qwen-segment-sft
#SBATCH --output=/home/ajenane/orena/segment_track/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/segment_track/logs/%x-%j.err

# DDP training for Qwen3.6-27B on the FOCUS segment track, 2x H200 on one node.
# Each GPU holds a full 27B+LoRA replica; gradients averaged every step.
#
# Effective batch = batch-size x grad-accum x n_gpu = 1 x 16 x 2 = 32, matching
# the frame runs so the convergence estimate carries over. Batch 1 measured FASTER
# than batch 2 (66s vs 71s/step) once the fla kernels removed the launch-overhead
# bottleneck -- larger per-device batches only add memory pressure now.
#
# 2-day limit: at N=80 an epoch is ~8.6 h, so 3 epochs (~26 h) no longer fits a
# 23 h wall. Early stopping on eval_loss is what actually ends the run.
#
# The 32 CPUs are load-bearing: each sample reads 64 JPEGs off NFS and that path
# is latency-bound (~71 ms/frame cold), so it only disappears behind compute with
# many dataloader workers. See segment_track/plan.md §2.6.
#
#   sbatch segment_track/sft_train_qwen_segment_ddp.slurm
#   sbatch --export=ALL,EXTRA_ARGS="--max-steps 20 --run-name smoke" \
#       segment_track/sft_train_qwen_segment_ddp.slurm

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SEG_DIR="$REPO_ROOT/segment_track"
mkdir -p "$SEG_DIR/logs"

# venv3.12 carries flash-linear-attention, which supplies the fused Gated DeltaNet
# kernels for the 48 linear-attention layers. Without it transformers falls back to
# a Python loop (~580k tiny kernel launches/step) and training is CPU-launch-bound.
# `.venv` is Python 3.13, for which flash-linear-attention has no working build.
VENV="${VENV:-$REPO_ROOT/venv3.12}"

TRAIN_FILE="${TRAIN_FILE:-$SEG_DIR/sft_export/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-$SEG_DIR/sft_export/eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

OUTPUT_DIR_ARGS=()
if [[ -n "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR_ARGS=(--output-dir "$OUTPUT_DIR")
fi

# NCCL over NVLink/PCIe on a single node; disable IB probing to avoid stalls.
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

echo "venv: $VENV"
"$VENV/bin/torchrun" --standalone --nproc_per_node=2 \
    "$SEG_DIR/sft_train_qwen_segment_ddp.py" \
    --train-file "$TRAIN_FILE" \
    --eval-file "$EVAL_FILE" \
    "${OUTPUT_DIR_ARGS[@]}" \
    $EXTRA_ARGS
