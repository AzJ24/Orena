#!/bin/bash
# Setup cost with the .sif on LOCAL disk — what the platform actually does.
#
# The platform pulls the image to the node before your process starts, and that
# pull is explicitly NOT charged to the latency budget. Running the .sif straight
# off NFS home therefore measures something the platform never pays: 44 GB of
# squashfs decompressed over NFS. The frame track measured the same gap
# (169.4 s NFS -> 152.5 s local -> 145.8 s warm), so this is the honest number.
#
#     sbatch orena_submission/segment-algorithm/test_localdisk.slurm
#
#SBATCH --job-name=seg-localdisk
#SBATCH --partition=gpu-large
#SBATCH --nodelist=gpu38
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/seg-localdisk-%j.out
#SBATCH --error=logs/seg-localdisk-%j.err
set -uo pipefail

SUB=$HOME/orena/orena_submission/segment-algorithm
SRC=$HOME/images/segment-algorithm-1.0.sif
LOCAL=/scratch/$USER-seg.sif
BATCH=$SUB/test/real_batch                 # 20 real questions, not the 3 samples
IN=$BATCH/interface_1
OUT=$BATCH/output_local

df -h /scratch | tail -1
echo "### staging .sif to local disk (mimics the platform's image pull; NOT charged) ###"
time cp "$SRC" "$LOCAL"
TMP=$(mktemp -d /scratch/$USER-segtmp-XXXX)
trap 'rm -rf "$TMP" "$LOCAL"' EXIT
rm -rf "$OUT"; mkdir -p "$OUT"

echo
echo "### RUN FROM LOCAL DISK, VRAM capped to the platform's 96 GB ###"
START=$(date +%s.%N)
apptainer exec --nv --no-home --pwd /opt/app \
  --bind "$IN":/input:ro --bind "$OUT":/output --bind "$TMP":/tmp \
  "$LOCAL" bash -c 'export HOME=/home/user; exec python -c "
import torch, runpy
t = torch.cuda.get_device_properties(0).total_memory/1024**3
torch.cuda.set_per_process_memory_fraction(min(96/t, 1.0))
runpy.run_path(\"/opt/app/inference.py\", run_name=\"__main__\")
"'
RC=$?
END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)

echo
echo "### verdict (20 questions -> budget 120 + 20x15 = 420 s) ###"
$HOME/orena/venv-cu128/bin/python "$SUB/test/score_batch.py" \
    --batch "$BATCH" --output-subdir output_local --elapsed "$ELAPSED" --rc "$RC" --per-question 15 \
    2>/dev/null || echo "scoring skipped"
