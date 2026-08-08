#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-10:00:00
#SBATCH --job-name=count-res-eval
#SBATCH --output=/home/ajenane/orena/orena_sft/logs/%x-%j.out
#SBATCH --error=/home/ajenane/orena/orena_sft/logs/%x-%j.err

# Resolution probe: does forcing image upscaling (more visual tokens per clip)
# improve instance counting? Frames are natively 415-922k px^2, below the
# processor's resize band, so only a raised MINPX (min pixel AREA) changes
# anything: MINPX=2073600 gives ~2x linear upscale -> ~2000 tokens/frame vs
# ~400-510 at native. Batch is 8, not 16: 4x the visual tokens of eval_direct.
#
#   # SFT arm (primary readout)
#   sbatch --export=ALL,MODEL=orena_sft/checkpoints/combined-9b-8r-direct \
#       orena_count/eval_res.slurm
#   # base arm (control: not anchored to the training resolution)
#   sbatch orena_count/eval_res.slurm

set -euo pipefail

REPO_ROOT="/home/ajenane/orena"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
BASE="${BASE:-Qwen/Qwen3.5-9B}"
MINPX="${MINPX:-2073600}"
# Must match training. The 'direct' system prompt carries the FO definition
# (graspers/scissors/trocars are NOT foreign objects); evaluating without it
# measures a train/inference mismatch, not resolution.
PSTYLE="${PSTYLE:-direct}"
OUT="$REPO_ROOT/orena_count/results/res${MINPX}_${PSTYLE}_$(basename "$MODEL")"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

echo "Node: $(hostname) | model=$MODEL | min_pixels=$MINPX | style=$PSTYLE -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/orena_sft/evaluate_qwen_frame.py" \
    --checkpoint-dir "$MODEL" \
    --base-model-id "$BASE" \
    --datasets lapchole heico \
    --prompt-style "$PSTYLE" \
    --min-pixels "$MINPX" \
    --max-new-tokens 32 \
    --batch-size 8 \
    --output-dir "$OUT"
