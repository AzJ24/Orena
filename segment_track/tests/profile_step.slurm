#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0-01:00:00
#SBATCH --job-name=segment-profile
#SBATCH --output=/home/ajenane/orena/segment_track/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/segment_track/logs/%x-%j.err
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv
/home/ajenane/orena/.venv/bin/python /home/ajenane/orena/segment_track/tests/profile_step.py
