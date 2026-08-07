#!/bin/bash
# Dry-run the container's inference.py on a GPU, WITHOUT a container.
#
# Podman/Apptainer live on cpu34, not here, so this exercises the submission logic
# first and leaves only the packaging to be proven by test_apptainer.slurm. It runs
# the very same inference.py the image ships — paths come from the ORENA_* env vars
# the file already honours — against the template's sample batch. Use venv-cu128,
# which mirrors the image's stack (torch 2.11.0 + CUDA 12.8, no fla); other venvs
# are a different torch and will not reproduce the image's behaviour.
#
#     sbatch orena_submission/segment-algorithm/test_native.slurm
#
# What a pass proves: the re-sharded weights load, decord decodes the platform's
# 5 fps clips, clip_frames rebuilds the 80-frame input, and generation lands well
# inside the 15 s/question budget. What it does NOT prove: the image builds, or
# that CUDA 12.4 is right — test_apptainer.slurm covers those.
#
#SBATCH --job-name=seg-native-test
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/seg-native-test-%j.out
#SBATCH --error=logs/seg-native-test-%j.err

set -euo pipefail

REPO="${REPO:-$HOME/orena}"
SUB_DIR="$REPO/orena_submission/segment-algorithm"
VENV="${VENV:-$REPO/venv-cu128}"

mkdir -p "$SUB_DIR/logs" "$SUB_DIR/test/output/interface_1"
rm -f "$SUB_DIR/test/output/interface_1"/*.json

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

export ORENA_INPUT_PATH="$SUB_DIR/test/input/interface_1"
export ORENA_OUTPUT_PATH="$SUB_DIR/test/output/interface_1"
export ORENA_MODEL_PATH="$SUB_DIR/resources/segment_model"
export TOKENIZERS_PARALLELISM=false

time "$VENV/bin/python" "$SUB_DIR/inference.py"

echo
echo "=+= answer.json:"
cat "$ORENA_OUTPUT_PATH/answer.json"
