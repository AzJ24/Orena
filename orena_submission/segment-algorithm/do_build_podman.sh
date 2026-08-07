#!/usr/bin/env bash
# Build the submission image with rootless Podman on the CPU build node (cpu34).
#
# The template's do_build.sh calls Docker, which this cluster does not have. Podman
# produces the same OCI/Docker image without root, so the artifact is identical --
# only the command differs. Layers land on cpu34's local disk under /localbuild,
# never on NFS home.
#
# Usage (on cpu34):
#     ./do_build_podman.sh
#     TAG=1.1 ./do_build_podman.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
IMAGE_TAG="segment-algorithm"
TAG="${TAG:-1.0}"

if ! command -v podman >/dev/null 2>&1; then
    echo "podman not found. Build on cpu34: ssh cpu34" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/resources/segment_model/model.safetensors.index.json" ]; then
    echo "FATAL: weights missing at $SCRIPT_DIR/resources/segment_model/" >&2
    echo "Generate them with reshard_weights.py — see SUBMISSION_NOTES.md" >&2
    exit 1
fi

echo "=+= Run ./preflight_deps.sh first if you have not — it proves the torch/CUDA"
echo "=+= stack in minutes, instead of after an hour of copying weights."
echo
echo "=+= Building ${IMAGE_TAG}:${TAG} (context: ${SCRIPT_DIR}, $(du -sh "$SCRIPT_DIR" | cut -f1))"
echo "=+= ~55 GB of weights are copied into the image; expect this to take a while."
echo "=+= Watch for: 'torch 2.11.0 | CUDA build 12.8' and 'PASSED' from verify_env.py"

# Podman requires fully-qualified image names; the Dockerfile uses the Docker-style
# short name the template ships. Rewrite only the FROM line into a build copy so the
# submitted Dockerfile stays faithful — the resulting image is identical.
BUILD_DIR=$(mktemp -d "/tmp/${USER}-segbuild-XXXXXX")
trap 'rm -rf "$BUILD_DIR"' EXIT
sed 's#^FROM --platform=linux/amd64 pytorch/pytorch:#FROM --platform=linux/amd64 docker.io/pytorch/pytorch:#' \
    "$SCRIPT_DIR/Dockerfile" > "$BUILD_DIR/Containerfile"
grep '^FROM' "$BUILD_DIR/Containerfile"

podman build \
    --platform=linux/amd64 \
    -f "$BUILD_DIR/Containerfile" \
    --tag "${IMAGE_TAG}:${TAG}" \
    "$SCRIPT_DIR"

echo
echo "=+= Built:"
podman images "${IMAGE_TAG}"
echo
echo "=+= Next: ./do_save_podman.sh   (upload archive + optional .sif for GPU testing)"
