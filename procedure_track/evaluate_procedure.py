"""Scores a procedure-track model with the FOCUS library's own Evaluator.

Forked from `segment_track/evaluate_base_segment.py`, which is the evaluator that
actually produced every segment number -- it shards, resumes, warms up and prefetches,
all of which matter more here because a prefix costs up to 10x what a clip did.

`Evaluator.run(track=Track.PROCEDURE)` enforces the 30 s ceiling by scoring slower
responses as WRONG rather than raising, so a latency regression looks like an accuracy
drop; the per-sample timings in predictions.jsonl are what tell the two apart.

Clips are built through `sampling`/`collate_procedure`, the same modules training uses,
so what the model sees here is byte-identical to what it was supervised on. The
`--frame-size`, `--frames-folder` and prompt arm MUST match the training run.

    # shard across GPUs
    for i in 0 1 2 3; do sbatch --export=ALL,SHARD=$i,NUM_SHARDS=4 \
        procedure_track/evaluate_procedure.slurm; done
    # then, once they finish
    venv3.12/bin/python procedure_track/evaluate_procedure.py --mode score
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

PROC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROC_DIR))
sys.path.insert(0, str(PROC_DIR.parent / "segment_track"))
sys.path.insert(0, str(PROC_DIR.parent / "orena_sft"))

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config  # noqa: E402
from focus.config import TRACK_MAX_LATENCY  # noqa: E402
from focus.data.data_models import Response  # noqa: E402
from focus.evaluation import Evaluator  # noqa: E402

from clip_sampling import frame_file  # noqa: E402

from collate_procedure import build_generation_inputs, remap_frame_dir  # noqa: E402
from prompts_procedure import build_system_prompt, extract_answer  # noqa: E402
from sampling import DEFAULT_FRAME_SIZE, to_native_index  # noqa: E402

DEFAULT_ROOT_DIR = Path("/projects/datasets_ML/orena/")
FORMATS = ["binary", "number", "percentage", "fo_class", "time",
           "multiple_choice", "open_ended"]
BUDGET = TRACK_MAX_LATENCY[Track.PROCEDURE]


def load_model(args):
    """Base model, or base + LoRA adapter merged down.

    Merging costs one pass at load time and removes the adapter indirection from every
    forward, so the latency figures are the ones a submission would see.
    """
    src = (args.checkpoint_dir
           if args.checkpoint_dir and (Path(args.checkpoint_dir) / "tokenizer_config.json").exists()
           else args.base_model_id)
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
        # interleave, so every shard sees the same mix of prefix lengths
        rows = rows[args.shard::args.num_shards]
    return rows


def warm_page_cache(record: dict, pool, frames_root: str | None = None) -> None:
    """Read the next prefix's JPEGs while the current one generates.

    Cold NFS is ~71 ms/frame, so 768 frames is ~55 s cold against a 30 s budget -- and
    it lands in `prep_time`, which the ceiling is judged on. Prefetching hides it so the
    recorded latency reflects the model rather than our filesystem. Note this assumes
    locally resident frames; a container reading a cold MP4 pays a different cost.
    """
    def _read(path):
        try:
            with open(path, "rb") as fh:
                fh.read()
        except OSError:
            pass

    for idx5 in record["frames_indices"]:
        pool.submit(_read, frame_file(remap_frame_dir(record["frame_dir"], frames_root),
                                      to_native_index(idx5, record["base_fps"])))


def generate_one(processor, model, record, system_prompt, frame_size, max_new_tokens,
                 frames_root=None):
    """One prefix -> one answer, timed the way the challenge times it: sampling and
    encoding the frames included, not just the forward pass."""
    total_start = time.monotonic()
    inputs = build_generation_inputs(processor, record, system_prompt, frame_size,
                                     frames_root).to(model.device)
    prep_time = time.monotonic() - total_start

    gen_start = time.monotonic()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generate_time = time.monotonic() - gen_start

    raw = processor.tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    answer = extract_answer(raw) if system_prompt else raw
    return answer, raw, prep_time, generate_time, time.monotonic() - total_start


def run_generation(dataset: str, records: list[dict], args) -> None:
    rows = select_rows(records, dataset, args)
    if not rows:
        print(f"[{dataset}] no rows selected; skipping.")
        return

    ds = FocusDataset(dataset, DatasetSplit.TEST, Track.PROCEDURE)
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
        done = {json.loads(line)["qID"] for line in pred_path.open()}
        rows = [r for r in rows if r["qID"] not in done]
        print(f"[{dataset}] resuming: {len(done)} already done, {len(rows)} left")

    nf = [r["n_frames"] for r in rows] or [0]
    print(f"[{dataset}] shard {args.shard}/{args.num_shards}: {len(rows)} prefixes "
          f"({min(nf)}-{max(nf)} frames each) -> {pred_path.name}")

    processor, model = args.processor, args.model
    if rows and not args.no_warmup:
        # The first call JIT-compiles the fla/Triton kernels (~30 s) and faults the
        # frames in over NFS. Untimed here, as it would be in a served model.
        t0 = time.monotonic()
        generate_one(processor, model, rows[0], args.system_prompt, args.frame_size,
                     args.max_new_tokens, args.frames_root)
        print(f"  [{dataset}/{args.shard}] warmup {time.monotonic() - t0:.1f}s (not recorded)")

    times = []
    pool = ThreadPoolExecutor(max_workers=args.prefetch_workers) if args.prefetch_workers else None
    if pool and rows:
        warm_page_cache(rows[0], pool, args.frames_root)
    with pred_path.open("a" if done else "w") as f:
        for i, record in enumerate(rows, 1):
            req, ref = by_qid[record["qID"]]
            if pool and i < len(rows):
                warm_page_cache(rows[i], pool, args.frames_root)
            answer, raw, prep_t, gen_t, total_t = generate_one(
                processor, model, record, args.system_prompt, args.frame_size,
                args.max_new_tokens, args.frames_root)
            times.append(total_t)

            f.write(json.dumps({
                "uid": record["uid"], "qID": req.qID, "videoID": record["videoID"],
                "procedure_type": record["procedure_type"],
                "format": ref._format, "primary_capability": ref.primary.name,
                "duration": record["duration"], "n_frames": record["n_frames"],
                "n_quoted_timestamps": record["n_quoted_timestamps"],
                "question": req.question, "gt_answer": ref.answer,
                "pred_answer": answer, "raw_response": raw,
                "prep_time": prep_t, "generate_time": gen_t, "total_time": total_t,
            }) + "\n")
            f.flush()

            if i % 25 == 0 or i == len(rows):
                over = sum(t > BUDGET for t in times)
                eta = statistics.median(times) * (len(rows) - i) / 3600
                print(f"  [{dataset}/{args.shard}] {i}/{len(rows)}  "
                      f"median {statistics.median(times):.2f}s  max {max(times):.2f}s  "
                      f"over-{BUDGET:.0f}s {over}  ETA {eta:.1f}h", flush=True)


def breakdown(rows: list[dict], key, label: str) -> None:
    """Accuracy proxy by an arbitrary slice.

    Uses exact string match, which under-counts the LLM-judged formats -- it is a shape
    check on the slices the Evaluator does not break out, not the headline number.
    """
    buckets: dict = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        b = buckets[key(r)]
        b[0] += str(r["pred_answer"]).strip().lower() == str(r["gt_answer"]).strip().lower()
        b[1] += 1
    print(f"  by {label}: " + "  ".join(
        f"{k}={c/n:.2f}(n={n})" for k, (c, n) in sorted(buckets.items(), key=lambda kv: str(kv[0]))))


def run_scoring(dataset: str, args) -> None:
    """Merge every shard for `dataset` and score once with the official Evaluator."""
    out_dir = Path(args.output_dir) / dataset
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

    ds = FocusDataset(dataset, DatasetSplit.TEST, Track.PROCEDURE)
    by_qid = {req.qID: (req, ref) for req, ref in ds}

    requests, references, responses = [], [], []
    for qid, row in merged.items():
        req, ref = by_qid[qid]
        requests.append(req)
        references.append(ref)
        responses.append(Response(qID=qid, content=row["pred_answer"], latency=row["total_time"]))

    with (out_dir / "predictions.all.jsonl").open("w") as f:
        for row in merged.values():
            f.write(json.dumps(row) + "\n")

    t = [r["total_time"] for r in merged.values()]
    print(f"[{dataset}] latency: median {statistics.median(t):.2f}s  "
          f"p95 {sorted(t)[int(0.95 * len(t))]:.2f}s  max {max(t):.2f}s  "
          f"over {BUDGET:.0f}s: {sum(x > BUDGET for x in t)}/{len(t)}")

    # Slices the Evaluator does not produce, and that this track specifically needs:
    # a 26-minute lapchole prefix and a 4-hour heico prefix are different problems.
    rows = list(merged.values())
    breakdown(rows, lambda r: f"{int(r['duration']//1800)*30}-{int(r['duration']//1800)*30+30}min",
              "prefix length")
    breakdown(rows, lambda r: "anchored" if r["n_quoted_timestamps"] else "unanchored",
              "question anchoring")

    evaluator = Evaluator(num_workers=args.judge_workers,
                          judge_kwargs={"device": args.judge_device})
    results_df, summary_df = evaluator.run(requests, references, responses,
                                           output_dir=out_dir, track=Track.PROCEDURE)
    n_slow = int(results_df["timed_out"].sum()) if "timed_out" in results_df else 0
    if n_slow:
        print(f"  !! {n_slow}/{len(results_df)} responses exceeded the {BUDGET:.0f}s "
              f"PROCEDURE limit and were scored WRONG")
    print(f"\n[{dataset}] summary:")
    print(summary_df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="LoRA adapter dir; omit to evaluate the untuned base model")
    ap.add_argument("--no-merge-lora", action="store_true")
    ap.add_argument("--test-file", default=str(PROC_DIR / "sft_export" / "test.jsonl"))
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"],
                    choices=["heico", "lapchole"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--output-dir", default=None,
                    help="defaults to <checkpoint-dir>/eval, or eval_base/<model name>")
    ap.add_argument("--mode", choices=["generate", "score", "all"], default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-warmup", action="store_true",
                    help="time the first prefix too, kernel compilation included")
    ap.add_argument("--fo-definitions", action="store_true")
    ap.add_argument("--system-prompt-file", type=Path, default=None)
    ap.add_argument("--frame-size", default=f"{DEFAULT_FRAME_SIZE[0]}x{DEFAULT_FRAME_SIZE[1]}",
                    help="MUST match training")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N per dataset")
    ap.add_argument("--formats", nargs="+", default=None, choices=FORMATS,
                    help="restrict to these answer formats. `--formats time` is the "
                         "timestamp probe: it is the only bucket that exercises the "
                         "clock end to end.")
    ap.add_argument("--prefetch-workers", type=int, default=16,
                    help="threads warming the next prefix's frames; 0 disables")
    # 1, not 4: the judge is itself a Qwen3.5 hybrid, so judging goes through
    # fla/Triton, whose autotuner is not thread-safe -- four workers crashed a clean
    # segment-track merge with "'NoneType' object is not a mapping".
    ap.add_argument("--frames-root", default=None,
                    help="re-root exported frame_dir paths onto this filesystem")
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
    else:
        args.system_prompt = build_system_prompt(args.fo_definitions, style="direct")

    if args.output_dir is None:
        args.output_dir = (Path(args.checkpoint_dir) / "eval" if args.checkpoint_dir
                           else PROC_DIR / "eval_base" / args.base_model_id.split("/")[-1])
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    set_config(FocusConfig(root_dir=args.root_dir))
    records = [json.loads(line) for line in Path(args.test_file).open()]
    print(f"test split: {len(records)} prefixes | output: {args.output_dir}")

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
