#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0-90:00:00
#SBATCH --job-name=qwen-frame-grpo
#SBATCH --output=/home/ajenane/orena/orena_grpo/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_grpo/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# GRPO training for Qwen3.5-9B on the FOCUS frame track (direct prompt), via
# orena_grpo/grpo.py. h200 -- the 9B policy plus G rollouts and the KL
# reference need the large card. All paths are absolute (anchored at REPO_ROOT)
# since #SBATCH directives and sbatch's working dir depend on where you invoke
# sbatch from, not where this script lives.
#
# Levers and hyperparameters go through EXTRA_ARGS (see grpo.py --help), e.g.:
#
#   # warm-start + KL, verifiable-only (no OpenAI key needed):
#   sbatch --export=ALL,EXTRA_ARGS="--init-from-sft --use-kl --skip-open-formats \
#     --group-size 10 --batch-size 20 --grad-accum 4" orena_grpo/grpo.slurm
#
#   # full run with the OpenAI judge for open-ended formats -- pass the key
#   # through the SAME --export token (no spaces after commas):
#   sbatch --export=ALL,OPENAI_API_KEY=sk-...,EXTRA_ARGS="--init-from-sft --use-kl \
#     --judge-model gpt-4o-mini --group-size 10 --batch-size 20 --grad-accum 4" \
#     orena_grpo/grpo.slurm
#
#   # cold-start ablation:
#   sbatch --export=ALL,EXTRA_ARGS="--no-init-from-sft --use-kl --skip-open-formats" \
#     orena_grpo/grpo.slurm

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
GRPO_DIR="$REPO_ROOT/orena_grpo"
mkdir -p "$GRPO_DIR/logs"

EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# OPENAI_API_KEY (if passed via --export) is inherited by the child process;
# grpo.py reads it from the environment for the open-ended judge.
"$REPO_ROOT/.venv/bin/python" "$GRPO_DIR/grpo.py" \
    $EXTRA_ARGS
