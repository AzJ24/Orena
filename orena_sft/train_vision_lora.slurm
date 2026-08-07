#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen-vision-lora
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# --lora-vision A/B against combined-9b-8r-direct. Hyperparameters are the ones
# recovered from that run's training_args.bin (single GPU, per-device batch 32,
# grad_accum 1, lr 1e-4, 3 epochs, eval/save every 50, seed 42) so the ONLY
# difference is the vision tower + patch merger getting LoRA adapters.
#
# The preflight runs first and the job aborts if the vision adapters receive no
# gradient -- that failure is silent at training time and would produce a null
# result for the wrong reason.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Batch 32 OOMs once backward flows through the vision tower (~2.9 GB/sample of
# activations vs 140 GB of card); 16 x 2 keeps the effective batch at 32.
export SMOKE_BATCH=16

echo "=== PREFLIGHT: vision LoRA gradient flow ==="
"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/preflight_vision_lora.py"
echo "=== PREFLIGHT PASSED -- starting training ==="

"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=1 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$SFT_DIR/sft_export/combined/train.jsonl" \
    --eval-file "$SFT_DIR/sft_export/combined/eval.jsonl" \
    --run-name combined-9b-8r-direct-vision \
    --prompt-style direct \
    --lora-vision \
    --lora-r 8 --lora-alpha 16 \
    --epochs 3 --lr 1e-4 --seed 42 \
    --batch-size 16 --grad-accum 2 \
    --eval-steps 50 --save-steps 50 \
    --early-stopping-patience 3
