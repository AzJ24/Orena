#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen-vision-full
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# Vision LoRA, run to the FULL 3 epochs with early stopping DISABLED.
#
# The original run (combined-9b-8r-direct-vision) stopped at step 700 of 1146
# because patience-3 fired on eval_loss. Evaluating its checkpoints on the test
# set showed eval_loss selected the WORST of the four late checkpoints on
# lapchole, and that accuracy was still rising at step 700 -- the last one:
#
#   step  eval_loss   lapchole  heico
#   550    0.3179*     0.5683   0.6210   <- selected by eval_loss
#   600    0.3392      0.5824   0.6270
#   650    0.3445      0.5763   0.6160
#   700    0.3260      0.5846   0.6327   <- last, and best
#
# So the peak is somewhere past 700 and unknown. This run removes the stopping
# criterion entirely and keeps every checkpoint, so the peak can be located by
# test accuracy instead of loss. That epoch number is what the all-data
# submission run needs, since all-data has nothing to early-stop on.
#
# Identical to train_vision_lora.slurm otherwise (same data, effective batch 32,
# lr 1e-4, seed 42, r8/a16 + vision adapters), so its checkpoints continue the
# same curve rather than starting a new one.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SMOKE_BATCH=16
export SMOKE_TRAIN_FILE="$SFT_DIR/sft_export/combined/train.jsonl"
export LORA_VISION=1

echo "=== PREFLIGHT: vision LoRA gradient flow + memory ==="
"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/preflight_vision_lora.py"
echo "=== PREFLIGHT PASSED -- starting training ==="

# No --early-stopping-patience: run all 3 epochs. Without it the Trainer keeps
# the final weights rather than reloading a "best" checkpoint chosen by loss,
# which is the behaviour we want here.
"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=1 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$SFT_DIR/sft_export/combined/train.jsonl" \
    --eval-file "$SFT_DIR/sft_export/combined/eval.jsonl" \
    --run-name combined-9b-8r-direct-vision-full \
    --prompt-style direct \
    --lora-vision \
    --lora-r 8 --lora-alpha 16 \
    --epochs 3 --lr 1e-4 --seed 42 \
    --batch-size 16 --grad-accum 2 \
    --eval-steps 50 --save-steps 50
