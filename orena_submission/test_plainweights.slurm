#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --nodelist=gpu38
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0-00:40:00
#SBATCH --job-name=fa-plain
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ajenane/orena/orena_submission/test_plain.out
#SBATCH --error=/home/ajenane/orena/orena_submission/test_plain.err
# Same .sif, same everything -- but the weights are bind-mounted as PLAIN FILES
# over the in-image squashfs path. This is what Docker/overlayfs gives you on
# the platform. Isolates squashfs decompression as the setup bottleneck.
set -x
SUB=/home/ajenane/orena/orena_submission
IN=$SUB/real_batch/interface_1
OUT=$SUB/real_batch/output_plain
WEIGHTS=$SUB/frame-algorithm/resources/frame_model
SIF=/home/ajenane/images/frame-algorithm-2026-07-27_09-45-04.sif
rm -rf "$OUT"; mkdir -p "$OUT"

START=$(date +%s.%N)
apptainer exec --nv --pwd /opt/app --no-home \
  --bind "$IN":/input:ro --bind "$OUT":/output \
  --bind "$WEIGHTS":/opt/app/resources/frame_model:ro \
  "$SIF" bash -c 'export HOME=/home/user; exec python inference.py' 2>&1 | \
  grep -E "Device:|setup took|Inference complete|inference done"
END=$(date +%s.%N)
echo "WALL CLOCK (plain-file weights): $(echo "$END - $START" | bc)s"
echo "   squashfs runs were: 169.4s (NFS) / 152.5s (local) / 145.8s (cached)"
