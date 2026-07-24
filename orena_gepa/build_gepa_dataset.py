"""Build small train/val JSONL sets for GEPA prompt optimization.

GEPA needs few examples -- rollouts are the expensive part, so a large set is
wasted budget. This samples a stratified handful from the FOCUS frame track:
`--n-train` questions to reflect on and `--n-val` to track the Pareto front.

Each record carries everything `metric.score_and_feedback` needs to score with
the *exact* official parser, including `format_kwargs` (e.g. the per-question
`threshold_seconds` for time answers) -- which the SFT export drops, and which
is why this does not just reuse `sft_export/*.jsonl`.

By default only deterministically-scorable formats are kept (binary, number,
fo_class, time, percentage) so the whole optimization runs with no LLM judge and
no API key. Pass `--include-judge-formats` to also sample open_ended / matching /
multiple_choice (then `run_gepa.py` must be given a judge).

Usage:
    .venv/bin/python orena_gepa/build_gepa_dataset.py --datasets heico lapchole
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config
from focus.config import DATASET_BASE_FPS
from focus.data.formats import JUDGE_FORMATS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orena_sft"))
from build_frame_sft_dataset import DEFAULT_ROOT_DIR, frame_path  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "gepa_data"


def load_records(dataset: str, split: DatasetSplit, cfg: FocusConfig, keep_judge: bool) -> list[dict]:
    base_fps = float(DATASET_BASE_FPS[dataset])
    ds = FocusDataset(dataset, split, Track.FRAME)
    records = []
    for req, ref in ds:
        if ref._format in JUDGE_FORMATS and not keep_judge:
            continue
        p = frame_path(cfg, dataset, base_fps, req.videoID, req.start_time)
        if not p.exists():
            continue
        records.append({
            "qID": req.qID,
            "source_dataset": dataset,
            "videoID": f"{dataset}/{req.videoID}",
            "procedure_type": req.procedure_type,
            "question": req.question,
            "answer": str(ref.answer),
            "format": ref._format,
            "format_kwargs": ref.format_kwargs,
            "primary_capability": ref.primary.name,
            "image_path": str(p),
            # kept so an LLM judge (judge formats) can be handed a faithful Request
            "start_time": req.start_time,
            "end_time": req.end_time,
        })
    return records


def stratified_sample(records: list[dict], n: int, seed: int, exclude_qids: set[str]) -> list[dict]:
    """Sample ~`n` records, balanced jointly across (answer format, capability),
    skipping `exclude_qids`.

    Stratifying on the (format, primary_capability) pair -- not format alone --
    keeps GEPA reflecting on the full spread of question *types*, so a rare
    capability isn't washed out by a common one that shares its answer format.
    Cells are filled round-robin up to a per-cell cap, so the budget is spread
    evenly rather than exhausted on the largest cells.
    """
    rng = np.random.RandomState(seed)
    pool = [r for r in records if r["qID"] not in exclude_qids]

    cells: dict[tuple[str, str], list[dict]] = {}
    for r in pool:
        cells.setdefault((r["format"], r["primary_capability"]), []).append(r)

    # Pre-shuffle each cell, then draw round-robin across cells until we hit n.
    shuffled = {}
    for key, rows in cells.items():
        order = rng.permutation(len(rows))
        shuffled[key] = [rows[i] for i in order]

    picked: list[dict] = []
    depth = 0
    max_depth = max((len(v) for v in shuffled.values()), default=0)
    while len(picked) < n and depth < max_depth:
        for key in sorted(shuffled):  # deterministic cell order
            if depth < len(shuffled[key]):
                picked.append(shuffled[key][depth])
                if len(picked) >= n:
                    break
        depth += 1

    rng.shuffle(picked)
    return picked[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"], choices=["heico", "lapchole"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--n-train", type=int, default=200, help="questions GEPA reflects on")
    ap.add_argument("--n-val", type=int, default=120, help="questions for Pareto tracking")
    ap.add_argument("--include-judge-formats", action="store_true",
                     help="also sample open_ended/matching/multiple_choice (needs a judge at run time)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    all_records: list[dict] = []
    for dataset in args.datasets:
        recs = load_records(dataset, DatasetSplit.TRAIN, cfg, args.include_judge_formats)
        print(f"  {dataset}: {len(recs)} usable frame-track questions")
        all_records += recs

    val = stratified_sample(all_records, args.n_val, args.seed, exclude_qids=set())
    train = stratified_sample(all_records, args.n_train, args.seed + 1,
                              exclude_qids={r["qID"] for r in val})

    fmt_counts, cap_counts = {}, {}
    for r in train + val:
        fmt_counts[r["format"]] = fmt_counts.get(r["format"], 0) + 1
        cap_counts[r["primary_capability"]] = cap_counts.get(r["primary_capability"], 0) + 1
    print(f"\nSampled {len(train)} train / {len(val)} val.")
    print(f"  format mix:     {fmt_counts}")
    print(f"  capability mix: {cap_counts}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
