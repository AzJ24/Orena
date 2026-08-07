#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=merge-lora
#SBATCH --output=/home/ajenane/orena/segment_track/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/segment_track/logs/%x-%j.err

# Folds a LoRA adapter into the base weights. ~55 GB of bf16 weights fit a single
# 96 GB card; the write is ~55 GB to /home.
set -euo pipefail
REPO_ROOT="/home/ajenane/orena"
ADAPTER="${ADAPTER:?set ADAPTER=<checkpoint dir>}"
OUT="${OUT:-}"
OUT_ARGS=()
[[ -n "$OUT" ]] && OUT_ARGS=(--out-dir "$OUT")
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$REPO_ROOT/venv3.12/bin/python" "$REPO_ROOT/segment_track/merge_lora.py" \
    --adapter-dir "$ADAPTER" "${OUT_ARGS[@]}"
