#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=0-08:00:00
#SBATCH --job-name=gepa-frame
#SBATCH --output=/home/ajenane/orena/orena_gepa/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_gepa/logs/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Evolves the FOCUS frame-track system prompt with GEPA (run_gepa.py).
# Task model = local VLM on the GPU; reflection LM = an API model via litellm.
# The reflection key is read from the repo-root .env by run_gepa.py, which maps
# an sk-or-... OPENAI_API_KEY to OPENROUTER_API_KEY -- so the default
# REFLECTION_LM below works with no extra env here.
#
# The default gres (rtx_pro_6000, 98GB) fits the 9B task model comfortably. For
# a 27B task model override to an h200, as base_model_eval.slurm documents:
#
#   sbatch --gres=gpu:h200:1 \
#     --export=ALL,MODEL_ID=Qwen/Qwen3.5-27B orena_gepa/run_gepa.slurm
#
# Override hyperparameters at submit time without editing this file, e.g. a
# quick smoke run:
#
#   sbatch --export=ALL,MAX_METRIC_CALLS=40 orena_gepa/run_gepa.slurm
#
# Full run: sbatch orena_gepa/run_gepa.slurm

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
GEPA_DIR="$REPO_ROOT/orena_gepa"
mkdir -p "$GEPA_DIR/logs"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
SEED_STYLE="${SEED_STYLE:-direct}"
REFLECTION_LM="${REFLECTION_LM:-openrouter/openai/gpt-4.1}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-32}"          # VLM rollout batch (task model throughput)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"  # answers are short; smaller = faster
DATASETS="${DATASETS:-heico lapchole}"
# Set FO_DEFINITIONS=1 to seed GEPA with the richer prompt that includes the
# per-class visual descriptions (the definitions the text-only reflection LM
# cannot regenerate). The out-dir is chosen per arm by run_gepa.py, so the two
# arms never overwrite each other's evolution log.
FO_DEFINITIONS="${FO_DEFINITIONS:-}"
# OUT_DIR left unset -> run_gepa.py derives runs/gepa-<seed>[-defs]-<refl>. Set it
# to keep separate runs (e.g. different budgets) from overwriting each other, and
# to avoid GEPA resuming from a stale checkpoint in an existing run_dir.
OUT_DIR="${OUT_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

DEFS_ARG=()
if [[ -n "$FO_DEFINITIONS" ]]; then
    DEFS_ARG=(--fo-definitions)
fi

OUT_DIR_ARGS=()
if [[ -n "$OUT_DIR" ]]; then
    OUT_DIR_ARGS=(--out-dir "$OUT_DIR")
fi

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Build the small optimization set once; reuse it on resubmits.
if [[ ! -f "$GEPA_DIR/gepa_data/train.jsonl" ]]; then
    "$REPO_ROOT/.venv/bin/python" "$GEPA_DIR/build_gepa_dataset.py" \
        --datasets $DATASETS --n-train 200 --n-val 120
fi

"$REPO_ROOT/.venv/bin/python" "$GEPA_DIR/run_gepa.py" \
    --model-id "$MODEL_ID" \
    --seed-style "$SEED_STYLE" \
    "${DEFS_ARG[@]}" \
    --reflection-lm "$REFLECTION_LM" \
    --max-metric-calls "$MAX_METRIC_CALLS" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${OUT_DIR_ARGS[@]}" \
    $EXTRA_ARGS
