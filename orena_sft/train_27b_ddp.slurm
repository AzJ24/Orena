#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen27b-frame-sft
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# Qwen3.6-27B vs the 9B baseline (combined-9b-8r-direct), same everything else:
# prompt-style direct, r=8/alpha=16, lr 1e-4, 3 epochs, eval/save every 50,
# seed 42, effective batch 32. Isolates base model size.
#
# 2x H200 DDP: 8 per device x 2 GPUs x grad-accum 2 = 32, matching the
# baseline's single-GPU batch 32 (and its 382 steps/epoch).
#
# Batch sizing comes from preflight_vision_lora.py against the LARGEST frames
# in the export, not the first N: 1280x720 lapchole frames carry 880 visual
# tokens vs 510 for heico, so a worst-case batch is ~1.7x the activations.
# Worst-case peak was 133.8 GiB at batch 16 -- only 6 GiB spare on a 140 GiB
# card, too tight for DDP. Batch 8 lands near 93 GiB.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# from_pretrained() resolves files against the Hub even when fully cached, and
# both DDP ranks do it at once -- a transient failed lookup surfaces as the
# misleading "does not appear to have a file named model-000NN-of-00015".
# The model is fully cached, so pin to cache-only.
export HF_HUB_OFFLINE=1

export SMOKE_MODEL_ID="Qwen/Qwen3.6-27B"
export SMOKE_BATCH="8"
export SMOKE_WORST_CASE=1

echo "=== PREFLIGHT: 27B at the training batch size, worst-case frames ==="
"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/preflight_vision_lora.py"
echo "=== PREFLIGHT PASSED -- starting training ==="

"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=2 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$SFT_DIR/sft_export/combined/train.jsonl" \
    --eval-file "$SFT_DIR/sft_export/combined/eval.jsonl" \
    --model-id "Qwen/Qwen3.6-27B" \
    --run-name combined-27b-8r-direct \
    --prompt-style direct \
    --lora-r 8 --lora-alpha 16 \
    --epochs 3 --lr 1e-4 --seed 42 \
    --batch-size 8 --grad-accum 2 \
    --eval-steps 50 --save-steps 50 \
    --early-stopping-patience 3
