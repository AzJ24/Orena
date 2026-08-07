"""Run inference.py with the fla fast path DISABLED, to price it.

The FRAME submission shipped on the template's base image with no
flash-linear-attention, hit the torch fallback, and still answered in ~0.25 s per
question. Whether that holds for SEGMENT is not obvious: the frame model has 24
linear-attention layers over ~1.5k tokens, this one has 48 over ~10k.

If the fallback meets the 15 s/question budget, `flash-linear-attention` can be
dropped from requirements.txt entirely — and with it the whole reason the segment
image needed a newer base than the template's.

`Qwen3_5GatedDeltaNet.__init__` resolves the kernels from module globals
(`chunk_gated_delta_rule or torch_chunk_gated_delta_rule`), so blanking those
globals BEFORE the model is constructed selects the fallback for every layer. This
isolates the kernel question from the torch-version question — the torch here is
whatever the calling venv has.

    python run_without_fla.py            # fallback
    python run_without_fla.py --with-fla # control, same script
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SUB = Path(__file__).resolve().parents[1]


def main() -> int:
    with_fla = "--with-fla" in sys.argv

    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    if not with_fla:
        m.chunk_gated_delta_rule = None
        m.fused_recurrent_gated_delta_rule = None
        m.causal_conv1d_fn = None
        m.causal_conv1d_update = None
        m.is_fast_path_available = False
    print(f"[fla] chunk_gated_delta_rule = "
          f"{'fla kernels' if m.chunk_gated_delta_rule else 'torch fallback'}", flush=True)

    import torch
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cap = float(sys.argv[sys.argv.index("--vram-gib") + 1]) if "--vram-gib" in sys.argv else 0
        if cap:
            torch.cuda.set_per_process_memory_fraction(min(cap / total, 1.0))
            print(f"[sim] VRAM capped to {cap:.0f} GiB of {total:.1f} GiB", flush=True)

    runpy.run_path(str(SUB / "inference.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
