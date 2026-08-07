#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --nodelist=gpu38
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0-00:50:00
#SBATCH --job-name=fa-localdisk
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ajenane/orena/orena_submission/test_localdisk.out
#SBATCH --error=/home/ajenane/orena/orena_submission/test_localdisk.err
set -x
SUB=/home/ajenane/orena/orena_submission
IN=$SUB/real_batch/interface_1
OUT=$SUB/real_batch/output_local
SRC=/home/ajenane/images/frame-algorithm-2026-07-27_09-45-04.sif
LOCAL=/scratch/$USER-fa.sif

df -h /scratch | tail -1
echo "### staging .sif to LOCAL disk (not charged; mimics platform pre-extracting the image) ###"
mkdir -p /scratch/$USER 2>/dev/null
time cp "$SRC" "$LOCAL"

rm -rf "$OUT"; mkdir -p "$OUT"
echo "### RUN FROM LOCAL DISK ###"
START=$(date +%s.%N)
apptainer exec --nv --pwd /opt/app --no-home --bind "$IN":/input:ro --bind "$OUT":/output \
  "$LOCAL" bash -c 'export HOME=/home/user; exec python inference.py' 2>&1 | \
  grep -E "Device:|setup took|Inference complete|inference done"
END=$(date +%s.%N)
echo "WALL CLOCK (local disk): $(echo "$END - $START" | bc)s   [NFS+squashfs run was 169.4s]"

echo "### second run: same file, now in page cache (upper bound on how fast I/O can get) ###"
rm -rf "$OUT"; mkdir -p "$OUT"
START=$(date +%s.%N)
apptainer exec --nv --pwd /opt/app --no-home --bind "$IN":/input:ro --bind "$OUT":/output \
  "$LOCAL" bash -c 'export HOME=/home/user; exec python inference.py' 2>&1 | \
  grep -E "setup took|inference done"
END=$(date +%s.%N)
echo "WALL CLOCK (warm cache): $(echo "$END - $START" | bc)s"
rm -f "$LOCAL"
