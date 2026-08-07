#!/bin/bash
#SBATCH --partition=gpu-large
#SBATCH --nodelist=gpu38
#SBATCH --gres=gpu:h200:1
#SBATCH --time=0-00:30:00
#SBATCH --job-name=fa-vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/ajenane/orena/orena_submission/test_vram.out
#SBATCH --error=/home/ajenane/orena/orena_submission/test_vram.err
SIF=/home/ajenane/images/frame-algorithm-2026-07-27_09-45-04.sif
SUB=/home/ajenane/orena/orena_submission
IN=$SUB/real_batch/interface_1
OUT=$SUB/real_batch/output_vram
rm -rf "$OUT"; mkdir -p "$OUT"
# poll VRAM every 0.5s for the whole run
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 0.5; done ) > /tmp/vram_poll.txt &
POLL=$!
apptainer exec --nv --pwd /opt/app --no-home --bind "$IN":/input:ro --bind "$OUT":/output \
  "$SIF" bash -c 'export HOME=/home/user; exec python inference.py' > /dev/null 2>&1
kill $POLL 2>/dev/null
echo "########## VRAM / GPU UTILISATION ##########"
python3 -c "
rows=[l.split(',') for l in open('/tmp/vram_poll.txt') if ',' in l]
mem=[int(r[0]) for r in rows]; util=[int(r[1]) for r in rows]
print(f'samples            : {len(mem)} @0.5s')
print(f'PEAK VRAM used     : {max(mem)} MiB  ({max(mem)/1024:.1f} GiB) of 143771 MiB (140.4 GiB)')
print(f'  as % of H200     : {max(mem)/143771*100:.1f}%')
print(f'  as % of L40S 48GB: {max(mem)/49140*100:.1f}%  <- platform GPU')
print(f'steady-state VRAM  : {sorted(mem)[len(mem)//2]} MiB')
print(f'peak GPU util      : {max(util)}%   mean {sum(util)/len(util):.0f}%')
"
