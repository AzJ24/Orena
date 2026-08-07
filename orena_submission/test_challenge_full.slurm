#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --nodelist=gpu38
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0-01:00:00
#SBATCH --job-name=fa-challenge
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ajenane/orena/orena_submission/test_challenge.out
#SBATCH --error=/home/ajenane/orena/orena_submission/test_challenge.err
#
# Full challenge-realistic run of the SHIPPED .sif:
#   - 20 real FOCUS TEST questions (10 heico + 10 lapchole), real frames
#   - /input read-only and /output bound exactly as the platform mounts them
#   - the WHOLE container run is timed, because the platform's clock starts at
#     process start (imports and model load count against the 120s setup share)
#   - budget = 120s setup + 20 x 5s = 220s; >20% over forfeits the whole batch
set -x

SIF=/home/ajenane/images/frame-algorithm-2026-07-27_09-45-04.sif
SUB=/home/ajenane/orena/orena_submission
IN=$SUB/real_batch/interface_1
OUT=$SUB/real_batch/output

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

rm -rf "$OUT"; mkdir -p "$OUT"

echo "########## FULL RUN (timed as the platform times it) ##########"
START=$(date +%s.%N)
apptainer exec --nv --pwd /opt/app --no-home \
  --bind "$IN":/input:ro \
  --bind "$OUT":/output \
  "$SIF" bash -c 'export HOME=/home/user; exec python inference.py'
RC=$?
END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)

echo "########## VERDICT ##########"
python3 - "$ELAPSED" "$RC" <<'PY'
import json, sys
elapsed, rc = float(sys.argv[1]), int(sys.argv[2])
IN  = "/home/ajenane/orena/orena_submission/real_batch/interface_1"
OUT = "/home/ajenane/orena/orena_submission/real_batch/output"

req = json.load(open(f"{IN}/request.json"))
B = len(req)
budget = 120 + B * 5
print(f"exit code            : {rc}")
print(f"questions in batch   : {B}")
print(f"WALL CLOCK (total)   : {elapsed:.1f}s")
print(f"BUDGET               : 120 + {B}x5 = {budget}s")
print(f"margin               : {budget - elapsed:.1f}s  ({elapsed/budget*100:.1f}% of budget used)")
print(f"forfeit threshold    : {budget*1.2:.0f}s (>20% over = whole batch lost)")
assert rc == 0, "FATAL: container exited non-zero"

ans = json.load(open(f"{OUT}/answer.json"))
print(f"\nresponses            : {len(ans)} / {B}")
assert len(ans) == B, "FATAL: response count mismatch"
assert {a['qID'] for a in ans} == {r['qID'] for r in req}, "FATAL: qID mismatch"
empty = [a['qID'] for a in ans if not str(a.get('content','')).strip()]
print(f"empty answers        : {len(empty)} {empty if empty else ''}")

qmap = {r['qID']: r['question'] for r in req}
print("\n--- answers ---")
for a in ans:
    print(f"  {a['qID']}  {a['content']!r:28}  <- {qmap[a['qID']][:64]}")

print()
if elapsed <= budget:
    print(f">>> PASS: {elapsed:.1f}s within {budget}s budget <<<")
else:
    print(f">>> FAIL: {elapsed:.1f}s EXCEEDS {budget}s budget <<<")
    sys.exit(1)
PY
