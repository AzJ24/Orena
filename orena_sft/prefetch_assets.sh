#!/bin/bash
# Run ONCE on a login node (compute nodes are usually offline) before
# train_27b_alldata_8gpu.slurm. Downloads everything that job needs from the Hub:
# the 27B weights (~54 GB) and the two FOCUS VQA datasets (annotations only --
# the frames come from DATA_ROOT on disk, not the Hub).
#
#   bash orena_sft/prefetch_assets.sh
#
# Set HF_HOME first if the default ~/.cache is on a small quota:
#   export HF_HOME=/mnt/vast/workspaces/VL_LeJepa/hf_cache

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"

echo "HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}"
df -h "${HF_HOME:-$HOME/.cache}" | tail -1

echo "=== 1/2  model weights: $MODEL (~54 GB) ==="
"$PY" - "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=[
    "*.safetensors", "*.json", "*.txt", "*.jinja", "*.py"])
print("cached at", p)
EOF

echo "=== 2/2  FOCUS annotations ==="
# Go through FocusDataset itself rather than calling load_dataset by hand: it
# loads a NAMED CONFIG ("frame"), and a data_dir= call caches under a different
# key, so the offline job would not find it.
"$PY" - <<'EOF'
from focus import DatasetSplit, FocusDataset, Track
for ds in ("heico", "lapchole"):
    for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
        d = FocusDataset(ds, split, Track.FRAME)
        print(f"  {ds} {split.value} (frame): {len(d)} rows")
EOF

echo
echo "Done. The training job can now run with HF_HUB_OFFLINE=1."
