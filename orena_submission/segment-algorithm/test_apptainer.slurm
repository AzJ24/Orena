#!/bin/bash
# Run the SHIPPED image the way the challenge platform runs it.
#
#     sbatch orena_submission/segment-algorithm/test_apptainer.slurm
#
# Simulated, deliberately:
#   * /input read-only, /output writable, /tmp a scratch volume — the platform's mounts
#   * no network (--net --network none), as the platform runs with networking disabled
#   * RTX PRO 6000 Blackwell forced (sm_120) — the GPU the challenge actually offers
#     for models this size. The L40S (48 GB) cannot hold this 55 GiB model, so
#     Blackwell is the only option, and CUDA 12.4 could not run on it at all
#   * VRAM capped to the platform's 96 GB via set_per_process_memory_fraction
#   * the WHOLE run timed, because the platform's clock starts at process start —
#     imports and the 55 GB model load count against the pooled budget
#
# Apptainer-only workarounds (Docker, and therefore the platform, needs none):
#   --no-home + HOME=/home/user : apptainer mounts the HOST home over the container's,
#                                 hiding the pip --user packages in /home/user/.local
#   --pwd /opt/app              : apptainer ignores the image's WORKDIR
#   apptainer exec ... python   : consequence of the two above; the ENTRYPOINT itself
#                                 is exercised by the platform, not here
#
#SBATCH --job-name=seg-sif-test
#SBATCH --partition=gpu-large
#SBATCH --nodelist=gpu38
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/seg-sif-test-%j.out
#SBATCH --error=logs/seg-sif-test-%j.err

set -uo pipefail

SUB_DIR="${SUB_DIR:-$HOME/orena/orena_submission/segment-algorithm}"
SIF="${SIF:-$HOME/images/segment-algorithm-1.0.sif}"
INPUT_DIR="${INPUT_DIR:-$SUB_DIR/test/input/interface_1}"
OUTPUT_DIR="${OUTPUT_DIR:-$SUB_DIR/test/output/interface_1}"
PLATFORM_VRAM_GIB="${PLATFORM_VRAM_GIB:-96}"   # the platform's RTX PRO 6000
PER_QUESTION_S="${PER_QUESTION_S:-15}"          # SEGMENT track budget

mkdir -p "$SUB_DIR/logs" "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*.json

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "SIF: $SIF"
[ -f "$SIF" ] || { echo "SIF not found — run do_save_podman.sh on cpu34 first" >&2; exit 1; }

# The image is read-only under apptainer, and TRITON_CACHE_DIR=/tmp/triton must be
# writable, so /tmp gets node-local scratch — mirroring the platform's /tmp volume.
TMP_SCRATCH=$(mktemp -d "/scratch/$USER-segtest-XXXXXX" 2>/dev/null \
    || mktemp -d "/tmp/$USER-segtest-XXXXXX")
trap 'rm -rf "$TMP_SCRATCH"' EXIT

RUN_ARGS=(--nv --no-home --pwd /opt/app
          --bind "$INPUT_DIR":/input:ro
          --bind "$OUTPUT_DIR":/output
          --bind "$TMP_SCRATCH":/tmp)

echo
echo "########## 1. dependency stack inside the shipped image ##########"
# The same assertions the build ran, now with the GPU checks that need a device.
# This is where a wrong-CUDA torch shows itself as cuda.is_available() == False.
apptainer exec "${RUN_ARGS[@]}" "$SIF" \
    bash -c 'export HOME=/home/user; exec python /opt/app/verify_env.py --gpu'
VERIFY_RC=$?
if [ $VERIFY_RC -ne 0 ]; then
    echo ">>> FATAL: dependency stack is wrong inside the image — do not submit <<<"
    exit 1
fi

echo
echo "########## 2. full batch, platform mounts, no network, VRAM capped to ${PLATFORM_VRAM_GIB} GiB ##########"
# --net --network none reproduces the platform's disabled networking: anything that
# lazily downloads a tokenizer or weights fails HERE rather than on the platform.
# Unprivileged network namespaces are not permitted in every apptainer.conf, so
# probe first and degrade to a warning rather than failing the whole test on it.
NET_ARGS=(--net --network none)
if ! apptainer exec "${NET_ARGS[@]}" "$SIF" true 2>/dev/null; then
    echo "WARN: unprivileged --net is not permitted here; running WITH network."
    echo "WARN: a lazy download would pass here and fail on the platform."
    NET_ARGS=()
fi

START=$(date +%s.%N)
apptainer exec "${NET_ARGS[@]}" "${RUN_ARGS[@]}" "$SIF" bash -c "
export HOME=/home/user
exec python -c \"
import torch, runpy
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
frac = min(${PLATFORM_VRAM_GIB} / total, 1.0)
torch.cuda.set_per_process_memory_fraction(frac)
print(f'[sim] capping VRAM to ${PLATFORM_VRAM_GIB} GiB of {total:.1f} GiB (fraction {frac:.3f})')
runpy.run_path('/opt/app/inference.py', run_name='__main__')
\""
RC=$?
END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)

echo
echo "########## 3. verdict ##########"
python3 - "$ELAPSED" "$RC" "$INPUT_DIR" "$OUTPUT_DIR" "$PER_QUESTION_S" <<'PY'
import json, sys
elapsed, rc, indir, outdir, per_q = (
    float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5]))

req = json.load(open(f"{indir}/request.json"))
B = len(req)
budget = 120 + B * per_q
print(f"exit code            : {rc}")
print(f"questions in batch   : {B}")
print(f"WALL CLOCK (total)   : {elapsed:.1f}s")
print(f"BUDGET               : 120 + {B}x{per_q:.0f} = {budget:.0f}s")
print(f"margin               : {budget - elapsed:.1f}s  ({elapsed / budget * 100:.1f}% used)")
print(f"forfeit threshold    : {budget * 1.2:.0f}s (>20% over = whole batch lost)")
if rc != 0:
    sys.exit("FATAL: container exited non-zero")

ans = json.load(open(f"{outdir}/answer.json"))
print(f"\nresponses            : {len(ans)} / {B}")
assert len(ans) == B, "FATAL: response count mismatch"
assert {a['qID'] for a in ans} == {r['qID'] for r in req}, "FATAL: qID mismatch"
empty = [a['qID'] for a in ans if not str(a.get('content', '')).strip()]
print(f"empty answers        : {len(empty)} {empty if empty else ''}")

qmap = {r['qID']: r['question'] for r in req}
print("\n--- answers ---")
for a in ans:
    print(f"  {a['qID']}  {a['content']!r:28}  <- {qmap[a['qID']][:64]}")

print()
if elapsed <= budget:
    print(f">>> PASS: {elapsed:.1f}s within {budget:.0f}s budget <<<")
else:
    print(f">>> FAIL: {elapsed:.1f}s EXCEEDS {budget:.0f}s budget <<<")
    sys.exit(1)
PY
