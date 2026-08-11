#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=eval-fp8-27b
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# Prices the FP8 quantisation penalty on the v2-split 27B, the only 27B
# checkpoint that is still locally measurable (bf16: lapchole 0.5666 /
# heico 0.6134). This gates the whole "27B under a 48 GB ceiling" route:
# bf16 27B is 51.0 GiB and does not fit, FP8 is 25.5 GiB and does.
#
# Runs on one RTX Pro 6000 (96 GB) so the H200s stay free for training.
#
#   sbatch orena_sft/eval_fp8_rtx.slurm              # FP8 arm
#   sbatch --export=ALL,FP8=0 orena_sft/eval_fp8_rtx.slurm   # bf16 control
#
# The bf16 control matters: the existing baseline was measured on an H200 at
# batch 8, and this runs on different hardware at a different batch. Comparing
# FP8-on-RTX against bf16-on-H200 would confound quantisation with both.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
CKPT="$REPO_ROOT/orena_sft/checkpoints/combined-27b-8r-direct-v2split"
FP8="${FP8:-1}"
BATCH="${BATCH:-16}"

if [[ "$FP8" == "1" ]]; then
    FP8_ARGS=(--fp8); OUT="$CKPT/eval_fp8"
else
    FP8_ARGS=(); OUT="$CKPT/eval_bf16_rtx"
fi

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

echo "Node: $(hostname) | fp8=$FP8 | batch=$BATCH -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/orena_sft/evaluate_qwen_frame.py" \
    --checkpoint-dir "$CKPT" \
    --base-model-id "Qwen/Qwen3.6-27B" \
    --prompt-style direct \
    --datasets lapchole heico \
    "${FP8_ARGS[@]}" \
    --max-new-tokens 32 \
    --batch-size "$BATCH" \
    --output-dir "$OUT"
