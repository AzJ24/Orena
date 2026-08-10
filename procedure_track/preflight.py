"""Pre-flight check before committing a multi-day procedure-track run.

Everything here is cheap and CPU-only. It exists because the expensive failures on this
pipeline are all silent: a missing kernel makes training 4x slower, an old Triton
computes wrong gradients, a stale export mismatches the sampler, a wrong `fps` puts
every timestamp 5x off, and a misconfigured save/eval cadence disables early stopping.
None of those announce themselves -- you find out hours in, or not at all.

    venv3.12/bin/python procedure_track/preflight.py
"""

from __future__ import annotations

import json
import shutil
import statistics
import sys
from importlib.util import find_spec
from pathlib import Path

PROC = Path(__file__).resolve().parent
sys.path.insert(0, str(PROC))
sys.path.insert(0, str(PROC.parent / "segment_track"))
sys.path.insert(0, str(PROC.parent / "orena_sft"))

FAIL, WARN = [], []


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        (WARN if warn_only else FAIL).append(label)
    return ok


print("1. environment / fused kernels")
import torch  # noqa: E402

print(f"      python {sys.version.split()[0]} | torch {torch.__version__}")
check("flash-linear-attention installed", find_spec("fla") is not None,
      "without it the linear-attention layers run a Python loop (4x slower)")
try:
    import triton

    v = tuple(int(x) for x in triton.__version__.split(".")[:3])
    check("triton >= 3.7.1", v >= (3, 7, 1),
          f"{triton.__version__} — below 3.7.1 computes WRONG gradients on Hopper (fla #640)")
except Exception as e:  # noqa: BLE001
    check("triton importable", False, str(e))
import transformers  # noqa: E402

check("transformers >= 5.9",
      tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (5, 9),
      transformers.__version__)
for mod in ("trl", "peft", "focus", "wandb", "decord", "cv2"):
    check(f"{mod} importable", find_spec(mod) is not None)

print("\n2. model weights present locally")
from huggingface_hub import snapshot_download  # noqa: E402

from sft_train_qwen_procedure_ddp import build_parser  # noqa: E402

a = vars(build_parser().parse_args([]))
try:
    p = snapshot_download(a["model_id"], local_files_only=True)
    shards = list(Path(p).glob("*.safetensors"))
    size = sum(f.stat().st_size for f in shards) / 1e9
    check(f"{a['model_id']} weights cached", bool(shards) and size > 10,
          f"{len(shards)} shards, {size:.0f} GB")
except Exception as e:  # noqa: BLE001
    check(f"{a['model_id']} weights cached", False, f"not downloaded ({type(e).__name__})")

print("\n3. dataset export")
from clip_sampling import frame_file  # noqa: E402

from sampling import DEFAULT_N_MAX, FPS, TARGET_PERIOD_S, to_native_index  # noqa: E402

splits = {}
for name in ("train", "eval", "test"):
    path = PROC / "sft_export" / f"{name}.jsonl"
    if not path.exists():
        check(f"{name}.jsonl exists", False, str(path))
        continue
    splits[name] = [json.loads(line) for line in path.open()]
    check(f"{name}.jsonl exists", True, f"{len(splits[name])} rows")

if len(splits) == 3:
    rows = [r for s in splits.values() for r in s]
    n = [r["n_frames"] for r in rows]
    check("frame counts are even (temporal_patch_size=2)", all(x % 2 == 0 for x in n))
    check(f"frame counts within --num-frames ({DEFAULT_N_MAX})", max(n) <= DEFAULT_N_MAX,
          f"max {max(n)}, median {statistics.median(n):.0f}")
    check("frame counts VARY (the capped-5s policy is active)", len(set(n)) > 1,
          f"{len(set(n))} distinct counts — a single value means flat-N sampling")
    at = [r["gap_at_target"] for r in rows]
    check("some rows reach true 5 s density", any(x >= 0.999 for x in at),
          f"{100*sum(x >= 0.999 for x in at)/len(at):.1f}% of rows fully at "
          f"{TARGET_PERIOD_S:.0f} s spacing")
    check("indices are monotone", all(
        all(b > a_ for a_, b in zip(r["frames_indices"], r["frames_indices"][1:]))
        for r in rows[:200]))

    vtr = {r["videoID"] for r in splits["train"]}
    check("train/eval video-disjoint", not (vtr & {r["videoID"] for r in splits["eval"]}))
    check("train/test video-disjoint", not (vtr & {r["videoID"] for r in splits["test"]}))

    s = splits["train"][0]
    check("frames resolve on disk",
          frame_file(s["frame_dir"], to_native_index(s["frames_indices"][0], s["base_fps"])).exists(),
          s["frame_dir"])
    check("frames folder is the overlay one", "frames_overlay" in s["frame_dir"],
          f"{s['frame_dir'].split('/')[-2]} — plan.md §2.3a wants the burned-in clock",
          warn_only=True)
    check("window line prepended to the question",
          s["messages"][0]["content"][1]["text"].startswith("Procedure observed from"),
          s["messages"][0]["content"][1]["text"][:48])
    check("timestamps are 5 fps absolute",
          abs(s["frames_indices"][-1] / FPS - s["end_time"]) < 2 * TARGET_PERIOD_S,
          f"last index {s['frames_indices'][-1]} -> {s['frames_indices'][-1]/FPS:.0f}s "
          f"vs window end {s['end_time']:.0f}s")

print("\n4. training configuration")
es, ss = a["eval_steps"], a["save_steps"]
check("save_steps is a multiple of eval_steps", ss % es == 0,
      f"save={ss} eval={es} — required for load_best_model_at_end")
check("early stopping configured", a["early_stopping_patience"] is not None,
      f"patience={a['early_stopping_patience']} evals")
check("gradient checkpointing on", not a["no_gradient_checkpointing"],
      "required at these sequence lengths")
check("LoRA r/alpha", (a["lora_r"], a["lora_alpha"]) == (8, 16),
      f"r={a['lora_r']} alpha={a['lora_alpha']}")
check("effective batch 32 on 2 GPUs", a["batch_size"] * a["grad_accum"] * 2 == 32,
      f"{a['batch_size']}x{a['grad_accum']}x2")

print("\n5. cost projection")
if "train" in splits:
    mean_frames = statistics.mean(r["n_frames"] for r in splits["train"])
    steps_per_epoch = len(splits["train"]) / 32
    # 81.7 s/step measured on the segment track at 8,800 video tokens with the 27B;
    # scaled by tokens and by the 9B's ~1/3 compute per token.
    step_s = 81.7 * (mean_frames * 110 / 8800) / 3.0
    train_h = steps_per_epoch * step_s / 3600
    print(f"      {steps_per_epoch:.0f} steps/epoch, mean {mean_frames:.0f} frames/row "
          f"-> est. {step_s:.0f} s/step")
    check("one epoch fits the 2-day SLURM limit", train_h < 44,
          f"est. {train_h:.1f} h (excludes eval)")
    free = shutil.disk_usage(PROC).free / 1e9
    check("disk space for checkpoints", free > 40, f"{free:.0f} GB free")

print("\n6. prompt")
from prompts_procedure import build_system_prompt, window_line  # noqa: E402

sp = build_system_prompt(style="direct")
check("procedure prompt builds", len(sp) > 1000, f"{len(sp)} chars")
check("prompt names both clocks", "<SECONDS seconds>" in sp and "burned into" in sp)
check("prompt states the window is a prefix", "nothing later exists" in sp)
check("prompt covers time + percentage", "hh:mm:ss" in sp and "percentage" in sp)
check("window_line formats correctly", window_line(10039) == "Procedure observed from "
      "00:00:00 to 02:47:19.", window_line(10039))

print("\n" + "=" * 70)
if FAIL:
    print(f"NOT READY — {len(FAIL)} blocking issue(s):")
    for f in FAIL:
        print(f"   - {f}")
elif WARN:
    print(f"READY (with {len(WARN)} warning(s)):")
    for w in WARN:
        print(f"   - {w}")
else:
    print("READY — all checks passed.")
sys.exit(1 if FAIL else 0)
