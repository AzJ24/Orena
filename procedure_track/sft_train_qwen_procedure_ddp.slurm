#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=qwen-procedure-sft
#SBATCH --output=/home/ajenane/orena/procedure_track/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/procedure_track/logs/%x-%j.err

# LoRA SFT for the FOCUS procedure track, 2 GPUs on one node.
#
# 2x RTX PRO 6000 (96 GB), not the H200s: the 9B needs far less than the 27B did, and
# leaving the H200s free matters more than the speed. Flip --gres to gpu:h200:2 if they
# are idle -- nothing else changes.
#
# Effective batch = batch-size x grad-accum x n_gpu = 1 x 16 x 2 = 32, matching both
# earlier tracks so their convergence estimates carry over.
#
# Sequence length is VARIABLE here: the capped-5s sampler gives a 26-minute lapchole
# prefix ~316 frames and a 4-hour heico prefix the full 768. Peak memory is set by the
# longest prefix in a batch, so an OOM will look intermittent rather than immediate.
# Escalation order is chunked cross-entropy -> --num-frames 512 at export -> frame size.
#
# The 32 CPUs are load-bearing: each sample reads up to 768 JPEGs off NFS and that path
# is latency-bound (~71 ms/frame cold). See procedure_track/plan.md §2.7.
#
#   sbatch --export=ALL,EXTRA_ARGS="--max-steps 20 --run-name smoke" \
#       procedure_track/sft_train_qwen_procedure_ddp.slurm
#   sbatch procedure_track/sft_train_qwen_procedure_ddp.slurm

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
PROC_DIR="$REPO_ROOT/procedure_track"
mkdir -p "$PROC_DIR/logs"

# venv3.12 carries flash-linear-attention, which supplies the fused Gated DeltaNet
# kernels. Without it transformers falls back to a Python loop over the linear-attention
# layers and training is CPU-launch-bound (measured 4x slower on the segment track).
# `.venv` is Python 3.13, for which flash-linear-attention has no working build.
VENV="${VENV:-$REPO_ROOT/venv3.12}"

TRAIN_FILE="${TRAIN_FILE:-$PROC_DIR/sft_export/train.jsonl}"
EVAL_FILE="${EVAL_FILE:-$PROC_DIR/sft_export/eval.jsonl}"
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
    "$PROC_DIR/sft_train_qwen_procedure_ddp.py" \
    --train-file "$TRAIN_FILE" \
    --eval-file "$EVAL_FILE" \
    "${OUTPUT_DIR_ARGS[@]}" \
    $EXTRA_ARGS
