#!/usr/bin/env bash
# Build the FRAME submission image with rootless podman on cpu34, then produce
# BOTH artifacts:
#   1. <name>.tar.gz  -- docker-archive, the file uploaded to the challenge platform
#   2. ~/images/<name>.sif -- for testing with apptainer on gpu38's H200
#
#   ssh cpu34
#   bash /home/ajenane/orena/orena_submission/build_on_cpu34.sh
set -euo pipefail

CTX=/home/ajenane/orena/orena_submission/frame-algorithm
TAG=frame-algorithm:1.0
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUT_DIR=/home/ajenane/orena/orena_submission
SIF=$HOME/images/frame-algorithm-${STAMP}.sif
TARGZ=$OUT_DIR/frame-algorithm_${STAMP}.tar.gz
LOG=$OUT_DIR/build_${STAMP}.log

# Mirror everything to a logfile in shared home, so the output survives an SSH
# drop and can be read from any node. The build takes 30-60+ min; run it under
# tmux/screen (or nohup) so a disconnect does not kill it.
mkdir -p "$OUT_DIR"
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"
echo "started $(date)"

echo "=== preflight ==="
command -v podman >/dev/null || { echo "podman not found -- are you on cpu34?"; exit 1; }
command -v apptainer >/dev/null || echo "WARN: apptainer not found; .sif step will be skipped"
[ -f "$CTX/resources/frame_model/model.safetensors" ] || {
    echo "FATAL: model weights missing at $CTX/resources/frame_model/"; exit 1; }
echo "context: $CTX ($(du -sh "$CTX" | cut -f1))"
df -h /localbuild 2>/dev/null | tail -1 || true

# Podman needs fully-qualified image names; the template's Dockerfile uses the
# Docker-style short name. Build from a temp copy so the submission Dockerfile
# stays byte-faithful to the template -- the resulting image is identical.
BUILD_DIR=$(mktemp -d /tmp/${USER}-fa-XXXX)
trap 'rm -rf "$BUILD_DIR"' EXIT
sed 's#^FROM --platform=linux/amd64 pytorch/pytorch:#FROM --platform=linux/amd64 docker.io/pytorch/pytorch:#' \
    "$CTX/Dockerfile" > "$BUILD_DIR/Containerfile"
echo "FROM line used:"; grep '^FROM' "$BUILD_DIR/Containerfile"

echo
echo "=== 1. podman build (watch for: torch 2.5.1 | CUDA 12.4) ==="
podman build --platform=linux/amd64 -f "$BUILD_DIR/Containerfile" -t "$TAG" "$CTX"

echo
echo "=== 2. submission artifact (docker-archive .tar.gz) ==="
podman save --format docker-archive "$TAG" | gzip > "$TARGZ"
ls -lh "$TARGZ"

echo
echo "=== 3. test artifact (.sif for apptainer on gpu38) ==="
if command -v apptainer >/dev/null; then
    mkdir -p "$HOME/images"
    podman save --format oci-archive -o "$BUILD_DIR/oci.tar" "$TAG"
    apptainer build "$SIF" "oci-archive:$BUILD_DIR/oci.tar"
    ls -lh "$SIF"
else
    echo "skipped (apptainer unavailable)"
fi

echo
echo "=== done $(date) ==="
echo "  submit : $TARGZ"
echo "  test   : $SIF"
echo "  log    : $LOG"
echo
echo "Next -- verify CUDA inside the image on an H200 (NOT the RTX PRO 6000,"
echo "which is Blackwell sm_120 and unsupported by this image's CUDA 12.4 torch):"
echo
echo "  srun -p gpu-large -w gpu38 --gres=gpu:h200:1 --time=00:20:00 \\"
echo "    apptainer exec --nv $SIF \\"
echo "    python -c \"import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),torch.cuda.get_device_name(0))\""
echo
echo "Then a full batch through the real entrypoint:"
echo "  srun -p gpu-large -w gpu38 --gres=gpu:h200:1 --time=00:30:00 \\"
echo "    apptainer run --nv \\"
echo "      --bind $CTX/test/input/interface_1:/input:ro \\"
echo "      --bind /tmp/fa-output:/output \\"
echo "      $SIF"
echo
echo "Cleanup once the .sif is verified:  podman rmi $TAG"
