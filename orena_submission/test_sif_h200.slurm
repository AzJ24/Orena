#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --nodelist=gpu38
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0-00:40:00
#SBATCH --job-name=fa-sif-h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ajenane/orena/orena_submission/test_sif_h200.out
#SBATCH --error=/home/ajenane/orena/orena_submission/test_sif_h200.err
#
# Verify the SHIPPED .sif on a real GPU. Uses an H200 (sm_90) -- NOT the
# RTX PRO 6000, which is Blackwell sm_120 and unsupported by this image's
# CUDA 12.4 torch, so it would fail for reasons irrelevant to the platform.
set -x

SIF=/home/ajenane/images/frame-algorithm-2026-07-27_09-45-04.sif
SUB=/home/ajenane/orena/orena_submission
OUT=$SUB/test_output/interface_1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "########## 1. CUDA inside the shipped image ##########"
apptainer exec --nv "$SIF" python -c "
import torch
print('torch       :', torch.__version__)
print('CUDA build  :', torch.version.cuda)
print('available   :', torch.cuda.is_available())
assert torch.cuda.is_available(), 'FATAL: CUDA NOT AVAILABLE INSIDE THE IMAGE'
print('device      :', torch.cuda.get_device_name(0))
print('capability  : sm_%d%d' % torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device='cuda')
print('matmul      :', float((x @ x).sum()))
print('>>> CUDA OK INSIDE IMAGE <<<')
"

echo "########## 2. full batch through the real entrypoint ##########"
# --pwd /opt/app is required: apptainer keeps the HOST cwd inside the container
# instead of honouring the image's WORKDIR, so the entrypoint's bare
# "python inference.py" would not resolve. Docker (and therefore the challenge
# platform) honours WORKDIR, so this flag is an apptainer-only test concern.
rm -rf "$OUT"; mkdir -p "$OUT"
# --no-home + HOME=/home/user are also apptainer-only: it mounts the HOST home
# over the container's by default, hiding the pip --user packages the Dockerfile
# installs into /home/user/.local. Docker does not do this, which is why the
# previous submission imported focus/torch/transformers fine on the platform.
# apptainer refuses --env HOME, so set it inside the container and invoke the
# entrypoint explicitly. Equivalent to what Docker does natively via
# USER user / WORKDIR /opt/app / ENTRYPOINT ["python","inference.py"].
time apptainer exec --nv --pwd /opt/app --no-home \
  --bind "$SUB/frame-algorithm/test/input/interface_1":/input:ro \
  --bind "$OUT":/output \
  "$SIF" bash -c 'export HOME=/home/user; exec python inference.py'

echo "########## 3. answer.json ##########"
cat "$OUT/answer.json"
python3 -c "
import json
a = json.load(open('$OUT/answer.json'))
r = json.load(open('$SUB/frame-algorithm/test/input/interface_1/request.json'))
print()
print('responses:', len(a), '| requests:', len(r))
assert len(a) == len(r), 'FATAL: response count mismatch'
assert {x['qID'] for x in a} == {x['qID'] for x in r}, 'FATAL: qID mismatch'
for x in a: print(f\"  {x['qID']}: {x['content']!r}  ({x['latency']:.2f}s)\")
print('>>> OUTPUT CONTRACT OK <<<')
"
