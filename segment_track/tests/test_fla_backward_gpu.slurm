#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=fla-backward
#SBATCH --output=/home/ajenane/orena/segment_track/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/segment_track/logs/%x-%j.err

# Validates fla's gradients on RTX PRO 6000 before committing a 9B run there.
set -euo pipefail
REPO_ROOT="/home/ajenane/orena"
"$REPO_ROOT/venv3.12/bin/python" "$REPO_ROOT/segment_track/tests/test_fla_backward_gpu.py"
