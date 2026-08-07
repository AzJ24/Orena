#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen-linear-attn-lora
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# --lora-linear-attn A/B against combined-9b-8r-direct. The default target
# modules reach only 8 of 32 LM layers (the rest use linear_attn); this arm
# adapts all 32. Vision tower stays frozen, so the ONLY difference from the
# baseline is the 120 extra linear_attn modules.
#
# Hyperparameters are those recovered from the baseline's training_args.bin.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LORA_LINEAR_ATTN=1
export SMOKE_BATCH=32

echo "=== PREFLIGHT: linear_attn LoRA gradient flow ==="
"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/preflight_vision_lora.py"
echo "=== PREFLIGHT PASSED -- starting training ==="

"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=1 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$SFT_DIR/sft_export/combined/train.jsonl" \
    --eval-file "$SFT_DIR/sft_export/combined/eval.jsonl" \
    --run-name combined-9b-8r-direct-linattn \
    --prompt-style direct \
    --lora-linear-attn \
    --lora-r 8 --lora-alpha 16 \
    --epochs 3 --lr 1e-4 --seed 42 \
    --batch-size 32 --grad-accum 1 \
    --eval-steps 50 --save-steps 50 \
    --early-stopping-patience 3
