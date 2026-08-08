#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --job-name=count-prompt-eval
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# Stage 1 of the enumerate-then-count test: prompt only, no training.
#
#   sbatch --export=ALL,PROMPT=enumerate_v1 orena_count/eval_prompt.slurm
#   sbatch --export=ALL,PROMPT=control_v1   orena_count/eval_prompt.slurm
#
# MODEL defaults to the base 9B (no adapter): evaluate_qwen_frame.py treats a
# checkpoint dir without adapter_config.json as a full model, so an HF id works.
#
# max-new-tokens must be set explicitly -- it defaults to 32, which truncates an
# enumeration mid-list and would score the prompt on parse failures.

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
PROMPT="${PROMPT:?set PROMPT=<name of a file in orena_count/prompts>}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
# BASE differs from MODEL when MODEL is a LoRA checkpoint dir; MAXTOK must be 32
# to match how eval_direct was run (160 only for prompts that emit a list).
BASE="${BASE:-$MODEL}"
MAXTOK="${MAXTOK:-160}"
OUT="$REPO_ROOT/orena_count/results/${PROMPT}_$(basename "$MODEL")"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

echo "Node: $(hostname) | model=$MODEL | prompt=$PROMPT -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/orena_sft/evaluate_qwen_frame.py" \
    --checkpoint-dir "$MODEL" \
    --base-model-id "$BASE" \
    --system-prompt-file "$REPO_ROOT/orena_count/prompts/${PROMPT}.txt" \
    --datasets lapchole heico \
    --max-new-tokens "$MAXTOK" \
    --batch-size 16 \
    --output-dir "$OUT"
