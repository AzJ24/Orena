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

echo "=== 2/2  FOCUS annotations (frame track) ==="
"$PY" - <<'EOF'
from datasets import load_dataset
for repo in ("orena-dkfz/heico-focus-vqa", "orena-dkfz/lapchole-focus-vqa"):
    for split in ("train", "test"):
        d = load_dataset(repo, data_dir="data/frame", split=split)
        print(f"  {repo} {split}: {len(d)} rows")
EOF

echo
echo "Done. The training job can now run with HF_HUB_OFFLINE=1."
