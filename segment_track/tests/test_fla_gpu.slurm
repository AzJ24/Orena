#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:20:00
#SBATCH --job-name=fla-gpu-test
#SBATCH --output=/home/ajenane/orena/segment_track/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/segment_track/logs/%x-%j.err
set -euo pipefail
/home/ajenane/orena/venv3.12/bin/python /home/ajenane/orena/segment_track/tests/test_fla_gpu.py
