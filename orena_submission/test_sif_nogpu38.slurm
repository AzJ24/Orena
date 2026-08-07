#!/bin/bash
#SBATCH --partition=gpu-unlimited
#SBATCH --gres=gpu:quadro_rtx_5000:1
#SBATCH --time=0-01:00:00
#SBATCH --job-name=fa-sif-verify
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/ajenane/orena/orena_submission/test_sif_verify.out
#SBATCH --error=/home/ajenane/orena/orena_submission/test_sif_verify.err
#
# Verify the shipped .sif without needing an H200 (both are held for up to 2 days).
#
#   Check 1 -- CUDA initialises INSIDE the image, on a Quadro RTX 5000 (sm_75,
#              supported by this image's CUDA 12.4 torch). This is THE regression
#              test: it is exactly what silently returned False on the platform.
#              Needs almost no VRAM, so 16 GB is plenty.
#   Check 2 -- full entrypoint end-to-end on CPU (no --nv), proving the container
#              reads /input, loads the checkpoint, generates, and writes a valid
#              answer.json. Slow (~3-4 min/question on CPU) but contract-complete.
#              The model is 18.8 GB so it cannot run on the GPU's 16 GB here.
set -x

SIF=/home/ajenane/images/frame-algorithm-2026-07-27_09-45-04.sif
SUB=/home/ajenane/orena/orena_submission
OUT=$SUB/test_output/interface_1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "########## CHECK 1: CUDA inside the shipped image ##########"
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
print('>>> CHECK 1 PASS: CUDA OK INSIDE IMAGE <<<')
"

echo "########## CHECK 2: full entrypoint on CPU ##########"
rm -rf "$OUT"; mkdir -p "$OUT"
time apptainer run \
  --bind "$SUB/frame-algorithm/test/input/interface_1":/input:ro \
  --bind "$OUT":/output \
  "$SIF"

echo "########## CHECK 3: answer.json contract ##########"
python3 -c "
import json
a = json.load(open('$OUT/answer.json'))
r = json.load(open('$SUB/frame-algorithm/test/input/interface_1/request.json'))
print('responses:', len(a), '| requests:', len(r))
assert len(a) == len(r), 'FATAL: response count mismatch'
assert {x['qID'] for x in a} == {x['qID'] for x in r}, 'FATAL: qID mismatch'
for x in a: print(f\"  {x['qID']}: {x['content']!r}\")
print('>>> CHECK 3 PASS: OUTPUT CONTRACT OK <<<')
"
