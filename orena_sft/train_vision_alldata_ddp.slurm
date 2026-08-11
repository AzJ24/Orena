#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=vision-alldata
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# SUBMISSION CANDIDATE -- 9B + vision LoRA on ALL 20000 labelled rows
# (train + eval + the public test split), 4 epochs, 2x H200 DDP.
#
# Config chosen from the checkpoint sweeps:
#   - vision LoRA is the best measured config (0.6091 / 0.6573 at epoch 3.01),
#     ahead of the 27B on both datasets, and fits 48 GB in bf16 (17.6 GiB).
#   - accuracy was still RISING at epoch 3 (lapchole +0.014 vs epoch 2.09,
#     p=0.022) with no OOD degradation (heico flat, p=0.96), so 4 epochs
#     extends past where any run has been measured.
#
# 20000 rows / effective batch 32 = 625 steps/epoch -> ~2500 steps for 4 epochs.
# DDP: 16 per device x 2 GPUs x accum 1 = 32, matching every run the epoch
# estimate came from. Vision adapters OOM at 32/device, hence 16.
#
# NO early stopping and NO eval-loss selection: eval loss picked the WORST of
# four late checkpoints on lapchole in the earlier vision run. With nothing to
# select on, every checkpoint from epoch 2 on is kept for post-hoc comparison:
# step 1250 (epoch 2) to 2500 (epoch 4) at --save-steps 50 is 26 checkpoints,
# so --save-total-limit 30 keeps them all (~6.6 GB) with margin.
#
# This checkpoint CANNOT be evaluated locally: every test row is training data.
# Judge it only on the submission platform.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
SFT_DIR="$REPO_ROOT/orena_sft"
mkdir -p "$SFT_DIR/logs"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN="$SFT_DIR/sft_export/combined_alldata_test/train.jsonl"
[[ -f "$TRAIN" ]] || {
    "$REPO_ROOT/.venv/bin/python" "$SFT_DIR/build_frame_sft_dataset.py" \
        --datasets heico lapchole --all-data --include-test \
        --out-dir "$SFT_DIR/sft_export/combined_alldata_test" \
        --manifest "$SFT_DIR/split_manifest_alldata_test.json"
}
echo "training rows: $(wc -l < "$TRAIN")"

# Nothing is held out; this slice is a divergence check only, NOT a selection
# signal (its rows are in the training set).
MONITOR="$SFT_DIR/sft_export/combined_alldata_test/monitor.jsonl"
[[ -f "$MONITOR" ]] || head -n 400 "$TRAIN" > "$MONITOR"

export SMOKE_BATCH=16
export SMOKE_TRAIN_FILE="$TRAIN"
export LORA_VISION=1
echo "=== PREFLIGHT: vision LoRA gradient flow + memory ==="
"$REPO_ROOT/.venv/bin/python" "$SFT_DIR/preflight_vision_lora.py"
echo "=== PREFLIGHT PASSED -- starting training ==="

"$REPO_ROOT/.venv/bin/torchrun" --standalone --nproc_per_node=2 \
    "$SFT_DIR/sft_train_qwen_frame_ddp.py" \
    --train-file "$TRAIN" \
    --eval-file "$MONITOR" \
    --run-name combined-9b-8r-vision-alldata-4ep \
    --prompt-style direct \
    --lora-vision \
    --lora-r 8 --lora-alpha 16 \
    --epochs 4 --lr 1e-4 --seed 42 \
    --batch-size 16 --grad-accum 1 \
    --eval-steps 250 --save-steps 50 --save-total-limit 30
