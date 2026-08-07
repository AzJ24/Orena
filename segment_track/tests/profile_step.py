"""Where does a training step actually go? Runs on one GPU, no DDP.

Answers three questions the step-time number alone cannot:
  1. Is the GPU busy at all, or are we waiting on data / CPU?
  2. Vision tower vs language model -- which dominates?
  3. Which CUDA kernels eat the time (i.e. is the linear-attention torch fallback
     really the culprit)?
"""

import json
import sys
import threading
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

SEG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEG))
sys.path.insert(0, str(SEG.parent / "orena_sft"))

from collate import build_collate_fn  # noqa: E402
from prompts import build_system_prompt  # noqa: E402

MODEL = "Qwen/Qwen3.6-27B"
N_SAMPLES = 3


class GpuSampler(threading.Thread):
    """Samples SM utilization every 50 ms; a low mean means we are not compute-bound."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.running = [], True

    def run(self):
        while self.running:
            try:
                self.samples.append(torch.cuda.utilization())
            except Exception:
                pass
            time.sleep(0.05)

    def stop(self):
        self.running = False
        self.join(timeout=2)
        return self.samples


def sync():
    torch.cuda.synchronize()


def main():
    rows = [json.loads(l) for l in (SEG / "sft_export" / "train.jsonl").open()][:N_SAMPLES]
    processor = AutoProcessor.from_pretrained(MODEL)
    system_prompt = build_system_prompt(style="direct", track="segment")
    collate = build_collate_fn(processor, system_prompt)

    print("=" * 70)
    print("A. DATA PATH (CPU, per sample)")
    t0 = time.monotonic()
    batches = [collate([r]) for r in rows]
    t_collate = (time.monotonic() - t0) / len(rows)
    print(f"   collate (64 JPEG reads + resize + processor): {t_collate:.2f} s/sample")
    b = batches[0]
    print(f"   seq len {b['input_ids'].shape[1]}, "
          f"video patches {tuple(b['pixel_values_videos'].shape)}, grid {b['video_grid_thw'].tolist()}")

    print("\nB. LOADING MODEL (bf16, LoRA r=8, gradient checkpointing)")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0})
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    inner = model.base_model.model          # unwrap PEFT
    dev = next(model.parameters()).device
    gpu = {k: v.to(dev) for k, v in b.items()}

    print("\nC. WARMUP (excluded from all timings)")
    out = model(**gpu)
    out.loss.backward()
    model.zero_grad(set_to_none=True)
    sync()

    print("\nD. COMPONENT TIMINGS (median of 3, GPU-synchronised)")

    def timeit(fn, n=3):
        ts = []
        for _ in range(n):
            sync(); t = time.monotonic()
            fn()
            sync(); ts.append(time.monotonic() - t)
        return sorted(ts)[n // 2]

    with torch.no_grad():
        t_vis = timeit(lambda: inner.get_video_features(
            gpu["pixel_values_videos"], gpu["video_grid_thw"]))
    print(f"   vision tower (forward only)      : {t_vis:6.2f} s")

    with torch.no_grad():
        t_fwd_nograd = timeit(lambda: model(**gpu))
    print(f"   full forward (no grad)           : {t_fwd_nograd:6.2f} s")
    print(f"     -> language model share        : {t_fwd_nograd - t_vis:6.2f} s")

    def fwd_only():
        model(**gpu)
    t_fwd = timeit(fwd_only)
    print(f"   full forward (grad enabled)      : {t_fwd:6.2f} s")

    def fwd_bwd():
        out = model(**gpu)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    t_step = timeit(fwd_bwd)
    print(f"   forward + backward (a micro-step): {t_step:6.2f} s")
    print(f"     -> backward + recompute        : {t_step - t_fwd:6.2f} s")

    print("\nE. GPU UTILIZATION during 3 forward+backward passes")
    s = GpuSampler(); s.start()
    for _ in range(3):
        fwd_bwd()
    sync()
    samples = s.stop()
    if samples:
        mean = sum(samples) / len(samples)
        busy = 100 * sum(1 for x in samples if x > 80) / len(samples)
        print(f"   mean SM utilization {mean:.0f}%  |  >80% for {busy:.0f}% of samples  "
              f"({len(samples)} samples)")

    print(f"\n   peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    print("\nF. TOP CUDA KERNELS (one forward+backward)")
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fwd_bwd()
        sync()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=18,
                                    max_name_column_width=55))

    print("\nG. ARITHMETIC")
    tok = b["input_ids"].shape[1]
    params = sum(p.numel() for p in model.parameters())
    tflop_fwd = 2 * params * tok / 1e12
    print(f"   {params/1e9:.1f}B params x {tok} tokens -> {tflop_fwd:.0f} TFLOP forward")
    print(f"   forward measured {t_fwd_nograd:.2f}s -> {tflop_fwd/t_fwd_nograd:.0f} TFLOPS "
          f"({100*tflop_fwd/t_fwd_nograd/990:.0f}% of H200 bf16 peak)")
    print(f"   step measured {t_step:.2f}s -> effective {3*tflop_fwd/t_step:.0f} TFLOPS "
          f"({100*3*tflop_fwd/t_step/990:.0f}% MFU)")


if __name__ == "__main__":
    main()
