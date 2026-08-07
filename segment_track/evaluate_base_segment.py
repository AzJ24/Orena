"""Baseline evaluation: the untuned Qwen3.6-27B on the SEGMENT test split.

Deliberately separate from `evaluate_qwen_segment.py` so that script stays
untouched while the training run uses it on checkpoints.

Two differences that matter beyond "no checkpoint":

  * Shardable. The test split is 6,254 clips at ~5 s each; one GPU is most of a
    day. `--shard i --num-shards n` interleaves the rows (so every shard gets the
    same mix of clip durations) and writes `predictions.shard{i}.jsonl`.
  * Split generate/score. Scoring needs every response for a dataset at once, so
    `--mode score` merges the shards and runs the Evaluator once.

    # 4 shards in parallel, one per GPU
    for i in 0 1 2 3; do sbatch --export=ALL,SHARD=$i,NUM_SHARDS=4 \
        segment_track/evaluate_base_segment.slurm; done
    # then, once they finish
    venv3.12/bin/python segment_track/evaluate_base_segment.py --mode score

`Evaluator.run(track=Track.SEGMENT)` enforces the 15 s ceiling: slower responses
are scored WRONG, not raised. The per-sample timings in predictions tell an
accuracy regression apart from a latency one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

SEG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SEG_DIR))
sys.path.insert(0, str(SEG_DIR.parent / "orena_sft"))

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config  # noqa: E402
from focus.data.data_models import Response  # noqa: E402
from focus.evaluation import Evaluator  # noqa: E402

from clip_sampling import DEFAULT_FRAME_SIZE, frame_file  # noqa: E402
from collate import build_generation_inputs  # noqa: E402
from prompts import build_system_prompt, extract_answer  # noqa: E402

DEFAULT_ROOT_DIR = Path("/projects/datasets_ML/orena/")
FORMATS = ["binary", "number", "percentage", "fo_class", "time",
           "multiple_choice", "open_ended"]


def load_model(args):
    """Base model, or base + LoRA adapter merged down.

    Merging costs one pass at load time and removes the adapter indirection from
    every forward, so the latency figures are the ones a submission would see.
    """
    src = args.checkpoint_dir if args.checkpoint_dir and \
        (Path(args.checkpoint_dir) / "tokenizer_config.json").exists() else args.base_model_id
    processor = AutoProcessor.from_pretrained(src)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model_id, dtype=torch.bfloat16, device_map="auto")
    if args.checkpoint_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.checkpoint_dir)
        if not args.no_merge_lora:
            model = model.merge_and_unload()
        print(f"loaded adapter from {args.checkpoint_dir} "
              f"({'merged' if not args.no_merge_lora else 'unmerged'})")
    return processor, model.eval()


def select_rows(records: list[dict], dataset: str, args) -> list[dict]:
    rows = [r for r in records if r["source_dataset"] == dataset]
    if args.formats:
        rows = [r for r in rows if r["format"] in args.formats]
    if args.limit:
        rows = rows[:args.limit]
    if args.num_shards > 1:
        rows = rows[args.shard::args.num_shards]
    return rows


def warm_page_cache(record: dict, pool) -> None:
    """Read the next clip's JPEGs while the current one generates.

    Measured on this cluster: 80 frames cost 0.48 s warm but 8.48 s cold over NFS,
    and that lands in `prep_time`, which the 15 s ceiling is judged on. Generation
    is a steady ~2.1 s, so an 8-thread prefetch hides the cold read entirely and
    the recorded latency reflects the model rather than our filesystem.
    """
    def _read(path):
        try:
            with open(path, "rb") as fh:
                fh.read()
        except OSError:
            pass

    for idx in record["frames_indices"]:
        pool.submit(_read, frame_file(record["frame_dir"], idx))


def generate_one(processor, model, record, system_prompt, frame_size, max_new_tokens):
    """One clip -> one answer, timed the way the challenge times it: clip
    loading and encoding included, not just the forward pass."""
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


def run_generation(dataset: str, records: list[dict], args) -> None:
    rows = select_rows(records, dataset, args)
    if not rows:
        print(f"[{dataset}] no rows selected; skipping.")
        return

    ds = FocusDataset(dataset, DatasetSplit.TEST, Track.SEGMENT)
    by_qid = {req.qID: (req, ref) for req, ref in ds}
    missing = [r["qID"] for r in rows if r["qID"] not in by_qid]
    if missing:
        raise SystemExit(f"{len(missing)} exported qIDs absent from FocusDataset "
                         f"(e.g. {missing[:3]}) -- export and library are out of sync.")

    out_dir = Path(args.output_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.num_shards == 1 else f".shard{args.shard}"
    pred_path = out_dir / f"predictions{suffix}.jsonl"

    done = set()
    if args.resume and pred_path.exists():
        done = {json.loads(l)["qID"] for l in pred_path.open()}
        rows = [r for r in rows if r["qID"] not in done]
        print(f"[{dataset}] resuming: {len(done)} already done, {len(rows)} left")

    print(f"[{dataset}] shard {args.shard}/{args.num_shards}: {len(rows)} clips "
          f"({len(records[0]['frames_indices'])} frames each) -> {pred_path.name}")

    processor, model = args.processor, args.model
    if rows and not args.no_warmup:
        # First call JIT-compiles the fla/Triton kernels (~30 s) and faults the
        # frames in over NFS. Untimed here, as it would be in a served model.
        t0 = time.monotonic()
        generate_one(processor, model, rows[0], args.system_prompt, args.frame_size,
                     args.max_new_tokens)
        print(f"  [{dataset}/{args.shard}] warmup {time.monotonic() - t0:.1f}s (not recorded)")

    times = []
    pool = ThreadPoolExecutor(max_workers=args.prefetch_workers) if args.prefetch_workers else None
    if pool and rows:
        warm_page_cache(rows[0], pool)
    with pred_path.open("a" if done else "w") as f:
        for i, record in enumerate(rows, 1):
            req, ref = by_qid[record["qID"]]
            if pool and i < len(rows):
                warm_page_cache(rows[i], pool)
            answer, raw, prep_t, gen_t, total_t = generate_one(
                processor, model, record, args.system_prompt, args.frame_size,
                args.max_new_tokens)
            times.append(total_t)

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

            if i % 25 == 0 or i == len(rows):
                over = sum(t > 15.0 for t in times)
                eta = statistics.median(times) * (len(rows) - i) / 3600
                print(f"  [{dataset}/{args.shard}] {i}/{len(rows)}  "
                      f"median {statistics.median(times):.2f}s  max {max(times):.2f}s  "
                      f"over-15s {over}  ETA {eta:.1f}h", flush=True)


def run_scoring(dataset: str, args) -> None:
    """Merge every shard for `dataset` and score once with the official Evaluator."""
    out_dir = Path(args.output_dir) / dataset
    # Only shard files, never `predictions.all.jsonl` (this function's own output)
    # nor a stale single-GPU `predictions.jsonl` from an earlier probe.
    shards = sorted(out_dir.glob("predictions.shard*.jsonl"))
    if not shards:
        solo = out_dir / "predictions.jsonl"
        shards = [solo] if solo.exists() else []
    if not shards:
        print(f"[{dataset}] nothing to score in {out_dir}")
        return

    merged: dict[str, dict] = {}
    for p in shards:
        for line in p.open():
            row = json.loads(line)
            merged[row["qID"]] = row
    print(f"[{dataset}] scoring {len(merged)} predictions from "
          f"{len(shards)} file(s): {[p.name for p in shards]}")

    ds = FocusDataset(dataset, DatasetSplit.TEST, Track.SEGMENT)
    by_qid = {req.qID: (req, ref) for req, ref in ds}

    requests, references, responses = [], [], []
    for qid, row in merged.items():
        req, ref = by_qid[qid]
        requests.append(req)
        references.append(ref)
        responses.append(Response(qID=qid, content=row["pred_answer"],
                                  latency=row["total_time"]))

    with (out_dir / "predictions.all.jsonl").open("w") as f:
        for row in merged.values():
            f.write(json.dumps(row) + "\n")

    t = [r["total_time"] for r in merged.values()]
    print(f"[{dataset}] latency: median {statistics.median(t):.2f}s  "
          f"p95 {sorted(t)[int(0.95 * len(t))]:.2f}s  max {max(t):.2f}s  "
          f"over 15 s: {sum(x > 15.0 for x in t)}/{len(t)}")

    evaluator = Evaluator(num_workers=args.judge_workers,
                          judge_kwargs={"device": args.judge_device})
    results_df, summary_df = evaluator.run(requests, references, responses,
                                           output_dir=out_dir, track=Track.SEGMENT)
    n_slow = int(results_df["timed_out"].sum()) if "timed_out" in results_df else 0
    if n_slow:
        print(f"  !! {n_slow}/{len(results_df)} responses exceeded the 15 s SEGMENT "
              f"limit and were scored WRONG")
    print(f"\n[{dataset}] summary:")
    print(summary_df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model-id", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="LoRA adapter dir; omit to evaluate the untuned base model")
    ap.add_argument("--no-merge-lora", action="store_true")
    ap.add_argument("--test-file", default=str(SEG_DIR / "sft_export" / "test.jsonl"))
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"],
                    choices=["heico", "lapchole"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--output-dir", default=None,
                    help="defaults to segment_track/eval_base/<model name>")
    ap.add_argument("--mode", choices=["generate", "score", "all"], default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="skip qIDs already present in this shard's predictions file")
    ap.add_argument("--no-warmup", action="store_true",
                    help="time the first clip too, kernel compilation included")
    ap.add_argument("--prompt-style", choices=["plain", "direct"], default="direct")
    ap.add_argument("--fo-definitions", action="store_true")
    ap.add_argument("--system-prompt-file", type=Path, default=None)
    ap.add_argument("--frame-size", default=f"{DEFAULT_FRAME_SIZE[0]}x{DEFAULT_FRAME_SIZE[1]}")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N per dataset")
    ap.add_argument("--formats", nargs="+", default=None, choices=FORMATS)
    ap.add_argument("--prefetch-workers", type=int, default=8,
                    help="threads warming the next clip's frames; 0 disables")
    # 1, not 4: the judge is itself a Qwen3.5 hybrid, so judging goes through
    # fla/Triton, whose autotuner is not thread-safe. Four workers crashed job
    # 12955 with "'NoneType' object is not a mapping" after a clean 4,000-row
    # merge. Judging ~460 open_ended rows serially costs ~90 s.
    ap.add_argument("--judge-workers", type=int, default=1)
    ap.add_argument("--judge-device", default="cuda")
    return ap


def main():
    args = build_parser().parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard must be in [0, {args.num_shards}), got {args.shard}")
    if args.mode == "score" and args.num_shards > 1:
        raise SystemExit("--mode score merges all shards; drop --num-shards")

    w, h = (int(x) for x in args.frame_size.lower().split("x"))
    args.frame_size = (w, h)

    if args.system_prompt_file is not None:
        args.system_prompt = args.system_prompt_file.read_text()
    elif args.prompt_style != "plain":
        args.system_prompt = build_system_prompt(args.fo_definitions,
                                                 style=args.prompt_style, track="segment")
    else:
        args.system_prompt = None

    if args.output_dir is None:
        args.output_dir = (Path(args.checkpoint_dir) / "eval" if args.checkpoint_dir
                           else SEG_DIR / "eval_base" / args.base_model_id.split("/")[-1])
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    set_config(FocusConfig(root_dir=args.root_dir))
    records = [json.loads(line) for line in Path(args.test_file).open()]
    print(f"test split: {len(records)} clips | output: {args.output_dir}")

    if args.mode in ("generate", "all"):
        args.processor, args.model = load_model(args)
        for dataset in args.datasets:
            run_generation(dataset, records, args)
        if args.mode == "all":
            del args.model
            torch.cuda.empty_cache()

    if args.mode in ("score", "all"):
        for dataset in args.datasets:
            run_scoring(dataset, args)


if __name__ == "__main__":
    main()
