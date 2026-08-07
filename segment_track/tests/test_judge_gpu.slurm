#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --job-name=judge-probe
#SBATCH --output=/home/ajenane/orena/segment_track/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/segment_track/logs/%x-%j.err
set -euo pipefail
"/home/ajenane/orena/venv3.12/bin/python" "/home/ajenane/orena/segment_track/tests/test_judge_gpu.py"
