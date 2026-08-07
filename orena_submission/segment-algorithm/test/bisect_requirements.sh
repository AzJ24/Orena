#!/usr/bin/env bash
# Which requirement drags torch off the template's base image?
#
# The FRAME submission ran on the template base (torch 2.5.1+cu124) with torch
# untouched, and worked on the platform. The SEGMENT requirements silently replaced
# it with torch 2.13.0+cu130 — a CUDA 13 wheel on a CUDA 12.4 image. This finds the
# exact line responsible, using `pip install --dry-run`, so nothing is downloaded and
# the venv is never mutated.
#
# Run against a venv holding ONLY the template base stack:
#     ./bisect_requirements.sh ~/orena/venv-cu124

set -uo pipefail
VENV="${1:-$HOME/orena/venv-cu124}"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"

echo "base stack: torch $("$PY" -c 'import torch;print(torch.__version__)')"
echo

# Cumulative sets: frame's proven set first, then each segment addition on top.
SETS=(
  "orena-focus==0.3.4 transformers==5.9.0 numpy>=1.23.0 pillow>=9.0|FRAME's exact set (proven on platform)"
  "orena-focus==0.3.4 transformers==5.9.0 numpy>=1.23.0 pillow>=9.0 decord==0.6.0|+ decord (segment needs video)"
  "orena-focus==0.3.5 transformers==5.9.0 numpy>=1.23.0 pillow>=9.0 decord==0.6.0|+ orena-focus 0.3.5"
  "orena-focus==0.3.4 transformers==5.9.0 numpy>=2.0 pillow>=9.0 decord==0.6.0|+ numpy>=2.0"
  "orena-focus==0.3.4 transformers==5.9.0 numpy>=1.23.0 pillow>=9.0 decord==0.6.0 accelerate==1.14.0|+ accelerate"
  "orena-focus==0.3.4 transformers==5.9.0 numpy>=1.23.0 pillow>=9.0 decord==0.6.0 flash-linear-attention==0.5.2|+ flash-linear-attention"
)

for entry in "${SETS[@]}"; do
    pkgs="${entry%%|*}"
    label="${entry##*|}"
    # --dry-run resolves fully but installs nothing, so each probe is independent.
    plan=$("$PIP" install --dry-run --quiet --report - $pkgs 2>/dev/null \
           | "$PY" -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('RESOLUTION-FAILED'); raise SystemExit
hits=[]
for it in d.get('install',[]):
    m=it['metadata']; n=m['name'].lower()
    if n in ('torch','torchvision','triton'):
        hits.append(f\"{m['name']}=={m['version']}\")
print(' '.join(hits) if hits else 'torch UNCHANGED')
")
    if [ "$plan" = "torch UNCHANGED" ]; then
        printf '  [ OK  ] %-48s %s\n' "$label" "$plan"
    else
        printf '  [BREAK] %-48s would install: %s\n' "$label" "$plan"
    fi
done

echo
echo "Any [BREAK] line replaces the base image's torch — on a CUDA 12.4 base that"
echo "means a CUDA 13 wheel, and a silent CPU fallback on an older driver."
