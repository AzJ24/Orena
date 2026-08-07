"""Evaluates a segment-track checkpoint with the FOCUS library's own Evaluator.

Deterministic parsing/comparison for closed-form formats (binary, number,
percentage, fo_class, time), LLM-as-judge for the rest -- and, because
`Evaluator.run(track=Track.SEGMENT)` enforces the track's 15 s latency limit,
responses slower than that are scored as WRONG rather than raising. A latency
regression therefore looks like an accuracy drop; the per-sample timings written
to predictions.jsonl are what tell the two apart.

Clips are built through `clip_sampling`/`collate`, the same modules training uses,
so what the model sees here is byte-identical to what it was supervised on. The
`--prompt-style` and `--frame-size` MUST match the training run.

Usage:
    .venv/bin/python segment_track/evaluate_qwen_segment.py \\
        --checkpoint-dir segment_track/checkpoints/<run> --datasets heico lapchole
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

SEG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SEG_DIR))
sys.path.insert(0, str(SEG_DIR.parent / "orena_sft"))

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config  # noqa: E402
from focus.data.data_models import Response  # noqa: E402
from focus.evaluation import Evaluator  # noqa: E402

from clip_sampling import DEFAULT_FRAME_SIZE  # noqa: E402
from collate import build_generation_inputs  # noqa: E402
from prompts import build_system_prompt, extract_answer  # noqa: E402

DEFAULT_ROOT_DIR = Path("/projects/datasets_ML/orena/")


def load_model(args):
    processor = AutoProcessor.from_pretrained(args.checkpoint_dir or args.base_model_id)
    if args.checkpoint_dir is None:
        # Baseline arm: the untuned base model, same prompt and same clips.
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.base_model_id, dtype=torch.bfloat16, device_map="auto")
        return processor, model

    adapter = Path(args.checkpoint_dir) / "adapter_config.json"
    if adapter.exists():
        from peft import PeftModel

        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.base_model_id, dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, args.checkpoint_dir)
        if not args.no_merge_lora:
            model = model.merge_and_unload()
    else:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.checkpoint_dir, dtype=torch.bfloat16, device_map="auto")
    return processor, model


def generate_one(processor, model, record, system_prompt, frame_size, max_new_tokens):
    """One clip -> one answer, with the timings the latency ceiling is judged on.

    Not batched: a segment clip is ~8k tokens, so a batch would multiply peak
    memory for little throughput, and the challenge serves one question at a time
    anyway -- timing one at a time is the honest measurement.
    """
    total_start = time.monotonic()
    inputs = build_generation_inputs(processor, record, system_prompt, frame_size).to(model.device)
    prep_time = time.monotonic() - total_start

    gen_start = time.monotonic()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generate_time = time.monotonic() - gen_start

    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    answer = extract_answer(raw) if system_prompt else raw
    return answer, raw, prep_time, generate_time, time.monotonic() - total_start


def evaluate_dataset(processor, model, dataset: str, records: list[dict], args) -> None:
    ds = FocusDataset(dataset, DatasetSplit.TEST, Track.SEGMENT)
    by_qid = {req.qID: (req, ref) for req, ref in ds}

    rows = [r for r in records if r["source_dataset"] == dataset]
    if args.formats:
        rows = [r for r in rows if r["format"] in args.formats]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print(f"[{dataset}] no rows match --formats {args.formats}; skipping.")
        return

    missing = [r["qID"] for r in rows if r["qID"] not in by_qid]
    if missing:
        raise SystemExit(f"{len(missing)} exported qIDs absent from FocusDataset "
                         f"(e.g. {missing[:3]}) -- export and library are out of sync.")

    out_dir = Path(args.output_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    print(f"\n[{dataset}] evaluating {len(rows)} segment clips "
          f"({len(rows[0]['frames_indices'])} frames each)...")

    requests, references, responses = [], [], []
    with pred_path.open("w") as f:
        for i, record in enumerate(rows, 1):
            req, ref = by_qid[record["qID"]]
            answer, raw, prep_t, gen_t, total_t = generate_one(
                processor, model, record, args.system_prompt, args.frame_size,
                args.max_new_tokens)

            requests.append(req)
            references.append(ref)
            # Latency judged by the challenge is the whole per-question cost:
            # sampling and encoding the clip, then generating.
            responses.append(Response(qID=req.qID, content=answer, latency=total_t))

            f.write(json.dumps({
                "uid": record["uid"], "qID": req.qID, "videoID": record["videoID"],
                "procedure_type": record["procedure_type"],
                "format": ref._format, "primary_capability": ref.primary.name,
                "duration": record["duration"],
                "question": req.question, "gt_answer": ref.answer,
                "pred_answer": answer, "raw_response": raw,
                "prep_time": prep_t, "generate_time": gen_t, "total_time": total_t,
            }) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(rows):
                print(f"  [{dataset}] {i}/{len(rows)}  (last total {total_t:.1f}s)")

    print(f"[{dataset}] wrote raw generations to {pred_path}")

    evaluator = Evaluator(num_workers=args.judge_workers,
                          judge_kwargs={"device": args.judge_device})
    # track=SEGMENT applies TRACK_MAX_LATENCY[SEGMENT] = 15 s; anything slower is
    # counted wrong and flagged `timed_out` in results.csv.
    results_df, summary_df = evaluator.run(requests, references, responses,
                                           output_dir=out_dir, track=Track.SEGMENT)

    n_slow = int(results_df["timed_out"].sum()) if "timed_out" in results_df else 0
    if n_slow:
        print(f"  !! {n_slow}/{len(results_df)} responses exceeded the 15 s SEGMENT "
              f"limit and were scored WRONG")
    print(f"\n[{dataset}] summary:")
    print(summary_df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="trained model dir; omit to evaluate the untuned base model")
    ap.add_argument("--base-model-id", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--no-merge-lora", action="store_true")
    ap.add_argument("--test-file", default=str(SEG_DIR / "sft_export" / "test.jsonl"))
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"],
                    choices=["heico", "lapchole"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--output-dir", default=None, help="defaults to <checkpoint-dir>/eval")
    ap.add_argument("--prompt-style", choices=["plain", "direct"], default="direct",
                    help="MUST match the style the checkpoint was trained with")
    ap.add_argument("--fo-definitions", action="store_true")
    ap.add_argument("--system-prompt-file", type=Path, default=None)
    ap.add_argument("--frame-size", default=f"{DEFAULT_FRAME_SIZE[0]}x{DEFAULT_FRAME_SIZE[1]}",
                    help="MUST match training")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N per dataset")
    ap.add_argument("--formats", nargs="+", default=None,
                    choices=["binary", "number", "percentage", "fo_class", "time",
                             "multiple_choice", "open_ended"],
                    help="restrict to these answer formats. `--formats time` is the "
                         "timestamp probe: it is the only bucket that exercises the "
                         "<seconds> markers and the hh:mm:ss conversion, and a random "
                         "sample of test rows can easily contain none of it.")
    ap.add_argument("--judge-workers", type=int, default=4)
    ap.add_argument("--judge-device", default="cuda")
    args = ap.parse_args()

    w, h = (int(x) for x in args.frame_size.lower().split("x"))
    args.frame_size = (w, h)

    if args.system_prompt_file is not None:
        args.system_prompt = args.system_prompt_file.read_text()
    elif args.prompt_style != "plain":
        args.system_prompt = build_system_prompt(args.fo_definitions, style=args.prompt_style,
                                                 track="segment")
    else:
        args.system_prompt = None

    if args.output_dir is None:
        args.output_dir = (Path(args.checkpoint_dir) / "eval" if args.checkpoint_dir
                           else SEG_DIR / "eval_base" / args.base_model_id.split("/")[-1])
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    set_config(FocusConfig(root_dir=args.root_dir))
    records = [json.loads(line) for line in Path(args.test_file).open()]

    processor, model = load_model(args)
    model.eval()

    for dataset in args.datasets:
        evaluate_dataset(processor, model, dataset, records, args)


if __name__ == "__main__":
    main()
