#!/usr/bin/env bash
# Prove the dependency stack BEFORE building the real image. Run this first.
#
# On the FRAME track a torch/CUDA mismatch was only discovered after the full image
# had been built, uploaded and scored — it does not raise, it just runs on CPU. This
# builds ONLY the Dockerfile's `deps` stage, from a context of two small files, so
# the same verification runs in a few minutes on a ~4 GB image instead of after an
# hour of copying 55 GB of weights.
#
# Passing here means: the torch in the image is still the base image's torch 2.11.0
# / CUDA 12.8 build, pip did not shadow it, flash-linear-attention did not sneak in
# (kept out by choice — this base ships triton 3.6.0, so it would now work),
# and transformers / orena-focus resolved to the versions the checkpoint was
# validated with. It does
# NOT prove CUDA works on a real device — there is no GPU on the build node.
# test_apptainer.slurm covers that.
#
# Usage (on cpu34):
#     ./preflight_deps.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
IMAGE="segment-deps-check"

if ! command -v podman >/dev/null 2>&1; then
    echo "podman not found. Run this on the build node: ssh cpu34" >&2
    exit 1
fi

# A minimal context: the build sends the context to the builder, and the real one
# is 55 GB of weights. `--target deps` never executes the weight COPYs, so they do
# not need to be present.
CTX=$(mktemp -d "/tmp/${USER}-segdeps-XXXXXX")
trap 'rm -rf "$CTX"' EXIT
cp "$SCRIPT_DIR/requirements.txt" "$SCRIPT_DIR/verify_env.py" "$CTX/"

# Podman requires fully-qualified image names; the Dockerfile uses the Docker-style
# short name, so rewrite just the FROM line into the build copy. The resulting image
# is identical — this keeps the submitted Dockerfile faithful to the template.
sed 's#^FROM --platform=linux/amd64 pytorch/pytorch:#FROM --platform=linux/amd64 docker.io/pytorch/pytorch:#' \
    "$SCRIPT_DIR/Dockerfile" > "$CTX/Containerfile"

echo "=+= Base image:"
grep '^FROM' "$CTX/Containerfile" | head -1
echo "=+= Context: $CTX ($(du -sh "$CTX" | cut -f1))"
echo

# verify_env.py --build-time runs inside the build (see the Dockerfile), so a bad
# resolution fails right here.
podman build \
    --platform=linux/amd64 \
    --target deps \
    -f "$CTX/Containerfile" \
    -t "$IMAGE" \
    "$CTX"

echo
echo "=+= Re-running the verification in the finished image (full output)"
podman run --rm --platform=linux/amd64 "$IMAGE" python /opt/app/verify_env.py --build-time

echo
echo "=+= Resolved package set (for the record)"
podman run --rm --platform=linux/amd64 "$IMAGE" python -m pip list --format=freeze \
    | grep -iE '^(torch|torchvision|triton|transformers|accelerate|fla|flash-linear|orena|decord|numpy|pillow)' || true

echo
echo "=+= PREFLIGHT PASSED — safe to run ./do_build_podman.sh"
echo "=+= Reclaim the check image with: podman rmi $IMAGE"
