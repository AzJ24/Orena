"""Answers one question with evidence: can a given torch/CUDA stack run this model?

Written to settle the base-image choice empirically rather than by argument. Point it
at a venv that mirrors a candidate base image and it reports, step by step, how far
the stack gets — imports, fla kernels, model load, a real generation.

    venv-cu124/bin/python .../probe_base_stack.py   # template's 2.5.1 / CUDA 12.4
    venv-cu126/bin/python .../probe_base_stack.py   # proposed 2.12.0 / CUDA 12.6

Each step prints PASS/FAIL and keeps going where it can, so a failure names the exact
incompatibility instead of just aborting.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]

results: list[tuple[str, bool, str]] = []


def step(label: str):
    def wrap(fn):
        try:
            detail = fn() or ""
            results.append((label, True, str(detail)))
            print(f"[PASS] {label}" + (f" — {detail}" if detail else ""))
            return True
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            results.append((label, False, msg))
            print(f"[FAIL] {label} — {msg}")
            if "-v" in sys.argv:
                traceback.print_exc()
            return False
    return wrap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=Path,
                    default=SUB / "resources" / "segment_model")
    ap.add_argument("--clip", type=Path, default=None,
                    help="an mp4 to run a real generation against")
    ap.add_argument("--skip-model", action="store_true",
                    help="imports only; no 55 GB load")
    args = ap.parse_args()

    print(f"python {sys.version.split()[0]}\n")

    import torch
    print(f"torch {torch.__version__} | CUDA build {torch.version.cuda}")
    try:
        import triton
        print(f"triton {triton.__version__}")
    except Exception:
        print("triton NOT INSTALLED")
    print()

    @step("torch.cuda.is_available()")
    def _cuda():
        assert torch.cuda.is_available(), "no CUDA device visible"
        return f"{torch.cuda.get_device_name(0)}, sm_%d%d" % torch.cuda.get_device_capability(0)

    @step("import transformers")
    def _tf():
        import transformers
        return f"transformers {transformers.__version__}"

    @step("Qwen3_5ForConditionalGeneration is available")
    def _cls():
        from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
        return "class importable"

    @step("fla Gated DeltaNet kernels import")
    def _fla():
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        assert chunk_gated_delta_rule is not None
        return "fla.ops.gated_delta_rule OK"

    @step("transformers resolved fla (not the slow torch fallback)")
    def _resolved():
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        assert m.chunk_gated_delta_rule is not None, \
            "transformers fell back to torch_chunk_gated_delta_rule"
        return "fast path wired up"

    @step("decord imports and decodes")
    def _decord():
        import decord
        if args.clip and args.clip.exists():
            vr = decord.VideoReader(str(args.clip), ctx=decord.cpu(0), num_threads=1)
            n, fps = len(vr), vr.get_avg_fps()
            del vr
            return f"decord {decord.__version__}, {n} frames @ {fps:.2f} fps"
        return f"decord {decord.__version__}"

    if args.skip_model:
        return report()

    @step("load the merged 27B checkpoint")
    def _load():
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        global processor, model
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model_path, dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else None)
        model.eval()
        return f"{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params, " \
               f"{torch.cuda.memory_allocated(0) / 1024**3:.1f} GiB"

    if results[-1][1] and args.clip and args.clip.exists():
        @step("generate an answer from a real clip")
        def _gen():
            sys.path.insert(0, str(SUB))
            import decord
            from resources.clip_frames import FRAME_SIZE, N_FRAMES, clip_inputs
            from resources.prompts import build_system_prompt, extract_answer

            reader = decord.VideoReader(str(args.clip), ctx=decord.cpu(0), num_threads=1)
            video, meta = clip_inputs(reader, 100.0, 129.0, FRAME_SIZE, N_FRAMES)
            del reader
            text = processor.apply_chat_template(
                [{"role": "system", "content": [{"type": "text", "text": build_system_prompt(
                    False, style="direct", track="segment")}]},
                 {"role": "user", "content": [
                     {"type": "video"},
                     {"type": "text", "text": "Is a foreign object visible in the scene?"}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = processor(text=[text], videos=[video], video_metadata=[meta],
                               do_sample_frames=False, return_tensors="pt").to(model.device)
            import time
            t0 = time.monotonic()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            dt = time.monotonic() - t0
            raw = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            return f"{extract_answer(raw)!r} in {dt:.2f}s ({inputs['input_ids'].shape[1]} tokens)"

    return report()


def report() -> int:
    failed = [label for label, ok, _ in results if not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"VERDICT: this stack CANNOT run the submission — {len(failed)} failure(s)")
        for label, ok, detail in results:
            if not ok:
                print(f"  - {label}: {detail}")
        return 1
    print("VERDICT: this stack can run the submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
