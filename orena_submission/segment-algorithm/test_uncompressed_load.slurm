#!/bin/bash
# Setup cost reading UNCOMPRESSED weights off local disk — what Docker actually does.
#
#     sbatch orena_submission/segment-algorithm/test_uncompressed_load.slurm
#
# Every apptainer measurement so far has been squashfs-bound, not I/O-bound:
# moving the .sif from NFS to local NVMe (8x the bandwidth) cut setup only 404 s ->
# 362 s, and both tracks decompress at the same ~145 MB/s (frame 18.8 GB/127 s,
# segment 51 GB/362 s) while the disk itself does 2.3 GB/s. Apptainer's SIF *is* a
# squashfs; a Docker/OCI image is tar layers extracted to overlayfs — ordinary
# uncompressed files. So the platform never pays that decompression, and the .sif
# numbers are a pessimistic upper bound.
#
# This measures the platform's actual situation: the same weights, uncompressed, on
# the same local NVMe. --mem=48G is deliberate — the 51 GB model cannot fit in a
# 48 GB cgroup page cache, so the staging copy cannot warm it and the read stays
# honest. inference.py's ORENA_MODEL_PATH override points it at the local copy.
#
#SBATCH --job-name=seg-uncompressed
#SBATCH --partition=gpu-large
#SBATCH --nodelist=gpu38
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/seg-uncompressed-%j.out
#SBATCH --error=logs/seg-uncompressed-%j.err
set -uo pipefail

SUB=$HOME/orena/orena_submission/segment-algorithm
SRC=$SUB/resources/segment_model
LOCAL=/scratch/$USER-segmodel
BATCH=$SUB/test/real_batch
OUT=$BATCH/output_uncompressed

df -h /scratch | tail -1
free -g | head -2

echo
echo "### staging UNCOMPRESSED weights to local NVMe (the platform's image pull) ###"
rm -rf "$LOCAL"
time cp -r "$SRC" "$LOCAL"
du -sh "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT

rm -rf "$OUT"; mkdir -p "$OUT"
export ORENA_INPUT_PATH="$BATCH/interface_1"
export ORENA_OUTPUT_PATH="$OUT"
export ORENA_MODEL_PATH="$LOCAL"
export TOKENIZERS_PARALLELISM=false

echo
echo "### RUN — 20 real questions, VRAM capped to the platform's 96 GB ###"
START=$(date +%s.%N)
$HOME/orena/venv-cu128/bin/python -c "
import torch, runpy
t = torch.cuda.get_device_properties(0).total_memory / 1024**3
torch.cuda.set_per_process_memory_fraction(min(96 / t, 1.0))
runpy.run_path('$SUB/inference.py', run_name='__main__')
"
RC=$?
END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)

echo
echo "### verdict ###"
$HOME/orena/venv-cu128/bin/python "$SUB/test/score_batch.py" \
    --batch "$BATCH" --output-subdir output_uncompressed \
    --elapsed "$ELAPSED" --rc "$RC" --per-question 15

echo
echo "### setup cost, all paths ###"
echo "  .sif squashfs / NFS       404 s"
echo "  .sif squashfs / NVMe      362 s"
echo "  raw weights / cold NFS    184 s"
echo "  raw weights / local NVMe  ^^^ see 'Setup complete' above  <- the platform's case"
