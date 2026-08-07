#!/usr/bin/env bash
# Produce the two artifacts, on cpu34, from the image do_build_podman.sh made:
#
#   1. <image>_<timestamp>.tar.gz  — docker-archive, the format the challenge
#      platform's "Upload a Container" expects. Identical to `docker save`'s output.
#   2. ~/images/<image>-<tag>.sif  — Apptainer's single-file format, only so the
#      image can be run on gpu38 for testing. NOT the submission artifact.
#
# Both intermediates are written to local disk (/localbuild or /tmp), never NFS:
# a 55 GB stream over NFS is slow and fills the home quota. Only the finished
# .tar.gz is moved to $OUT_DIR at the end.
#
# Usage (on cpu34):
#     ./do_save_podman.sh              # archive + .sif
#     SKIP_SIF=1 ./do_save_podman.sh   # archive only

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
IMAGE_TAG="segment-algorithm"
TAG="${TAG:-1.0}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR}"
SIF_DIR="${SIF_DIR:-$HOME/images}"

# Local scratch for the multi-gigabyte intermediates.
WORK="${WORK:-/localbuild/$(id -u)/save-$$}"
if [ ! -d "$(dirname "$WORK")" ]; then
    WORK="/tmp/$USER/save-$$"
fi
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

if ! podman image exists "${IMAGE_TAG}:${TAG}"; then
    echo "Image ${IMAGE_TAG}:${TAG} not found — run ./do_build_podman.sh first" >&2
    exit 1
fi

# podman and docker print .Created in DIFFERENT formats, and the template's
# docker-shaped sed silently no-ops on podman's, leaving the raw string — spaces,
# dots and all — in the filename:
#     docker : 2026-08-03T12:02:47.125Z
#     podman : 2026-08-03 12:02:47.12548048 +0000 UTC
# Normalise both to 2026-08-03_12-02-47: split on the first T-or-space, drop the
# fractional seconds and everything after, then colons to dashes. The tr guard
# strips anything unexpected rather than trusting the match, and an empty result
# falls back to the wall clock.
build_timestamp=$(podman inspect --format='{{ .Created }}' "${IMAGE_TAG}:${TAG}")
stamp=$(printf '%s' "$build_timestamp" | sed -E 's/[T ]/_/; s/\..*//; s/:/-/g' | tr -cd 'A-Za-z0-9._-')
[ -n "$stamp" ] || stamp=$(date +%Y-%m-%d_%H-%M-%S)
output_filename="${OUT_DIR}/${IMAGE_TAG}_${stamp}.tar.gz"

# pigz saturates the build node's cores; gzip alone on 55 GB is painfully slow.
if command -v pigz >/dev/null 2>&1; then compressor=(pigz -c); else compressor=(gzip -c); fi

echo "=+= [1/2] Saving docker-archive -> ${output_filename}"
podman save --format docker-archive -o "${WORK}/image.tar" "${IMAGE_TAG}:${TAG}"

mkdir -p "$OUT_DIR"
if command -v pv >/dev/null 2>&1; then
    pv "${WORK}/image.tar" | "${compressor[@]}" >"$output_filename"
else
    "${compressor[@]}" <"${WORK}/image.tar" >"$output_filename"
fi
rm -f "${WORK}/image.tar"

sha256sum "$output_filename" >"${output_filename}.sha256"
echo "=+= Upload artifact ready: $(du -h "$output_filename" | cut -f1)  $output_filename"

if [ -n "${SKIP_SIF:-}" ]; then
    echo "=+= SKIP_SIF set — not building a .sif"
    exit 0
fi

echo
echo "=+= [2/2] Converting to Apptainer .sif for GPU testing on gpu38"
if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer not found on this node — skipping the .sif" >&2
    exit 0
fi

mkdir -p "$SIF_DIR"

# apptainer EXTRACTS the whole OCI image before squashing it, so it needs ~62 GB of
# scratch on top of the ~62 GB output. Its default tmpdir is /tmp, which on a build
# node is typically far too small -- the symptom is mksquashfs dying with
# "Write failed because No space left on device" only after the extract succeeds.
# Point both scratch dirs at the same local disk the rest of this script uses.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$WORK/apptainer-tmp}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$WORK/apptainer-cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
echo "=+= apptainer scratch: $APPTAINER_TMPDIR"
df -h "$APPTAINER_TMPDIR" "$SIF_DIR" | sed 's/^/    /'

podman save --format oci-archive -o "${WORK}/image-oci.tar" "${IMAGE_TAG}:${TAG}"
apptainer build --force "${SIF_DIR}/${IMAGE_TAG}-${TAG}.sif" "oci-archive:${WORK}/image-oci.tar"

echo
echo "=+= Test image: ${SIF_DIR}/${IMAGE_TAG}-${TAG}.sif"
echo "=+= Test it:    sbatch ${SCRIPT_DIR}/test_apptainer.slurm"
echo "=+= Submit:     upload ${output_filename}"
