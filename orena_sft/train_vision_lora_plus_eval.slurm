#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen-vision-pluseval
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# A/B partner to train_vision_lora_full.slurm (job 13490). Identical in every
# respect except the training pool:
#
#   13490 (full):      12225 rows (train only)      382 steps/epoch
#   this  (+eval):     13748 rows (train + eval)    430 steps/epoch
#
# Once early stopping is gone the 1523 eval rows are no longer earning their
# keep as a selection signal, so this measures what folding them back into
# training actually buys. Test is excluded from both -- it stays the yardstick.
#
# COMPARE AT MATCHED EPOCHS, NOT STEPS: the two runs have different
# steps-per-epoch, so e.g. epoch 1.83 is step 700 in 13490 and step 787 here.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN="$SFT_DIR/sft_export/combined_alldata/train.jsonl"
[[ -f "$TRAIN" ]] || {
    echo "building train+eval export (test excluded, asserted)"
    "$REPO_ROOT/.venv/bin/python" "$SFT_DIR/build_frame_sft_dataset.py" \
        --datasets heico lapchole --all-data \
        --out-dir "$SFT_DIR/sft_export/combined_alldata" \
        --manifest "$SFT_DIR/split_manifest_alldata.json"
}

# Nothing is held out, so the Trainer's eval_dataset is a slice of train used
# only as a divergence check. Not a selection signal.
MONITOR="$SFT_DIR/sft_export/combined_alldata/monitor.jsonl"
[[ -f "$MONITOR" ]] || head -n 400 "$TRAIN" > "$MONITOR"

export SMOKE_BATCH=16
export SMOKE_TRAIN_FILE="$TRAIN"
export LORA_VISION=1
echo "=== PREFLIGHT: vision LoRA gradient flow + memory ==="
"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/preflight_vision_lora.py"
echo "=== PREFLIGHT PASSED -- starting training ==="

"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=1 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$TRAIN" \
    --eval-file "$MONITOR" \
    --run-name combined-9b-8r-direct-vision-pluseval \
    --prompt-style direct \
    --lora-vision \
    --lora-r 8 --lora-alpha 16 \
    --epochs 3 --lr 1e-4 --seed 42 \
    --batch-size 16 --grad-accum 2 \
    --eval-steps 50 --save-steps 50
