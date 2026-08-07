"""A fixed test subsample for sweeping many checkpoints affordably.

Scoring every checkpoint on all 6,254 clips is 1.26 h each; 2,000 rows is ~24 min.
Two design choices matter:

  * 1,000 rows per procedure_type, not proportional. The OOD (Sigmoid Resection)
    and ID (Lap Chole) halves are read as SEPARATE numbers, so equal allocation
    gives them equal precision -- proportional would leave the smaller ID half
    noisier for no benefit.
  * Within a procedure, proportional over (answer_format x clip_length) with a
    floor of 1, so each half stays an unbiased estimate of its own procedure and
    the weak `time` x 299 s cell is never empty.

Fixed seed: every checkpoint must be scored on the SAME rows or the curve
measures sampling noise instead of training progress.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

SEG = Path(__file__).resolve().parent


def bucket(d: float) -> int | str:
    r = round(d)
    return r if r in (29, 119, 299) else "other"


def allocate(cells: dict, total: int) -> dict:
    """Largest-remainder proportional allocation, floor 1, capped at cell size."""
    n = sum(len(v) for v in cells.values())
    raw = {k: len(v) * total / n for k, v in cells.items()}
    take = {k: min(len(cells[k]), max(1, int(raw[k]))) for k in cells}
    # distribute the remainder by largest fractional part, respecting cell sizes
    order = sorted(cells, key=lambda k: raw[k] - int(raw[k]), reverse=True)
    while sum(take.values()) < total:
        progressed = False
        for k in order:
            if sum(take.values()) >= total:
                break
            if take[k] < len(cells[k]):
                take[k] += 1
                progressed = True
        if not progressed:
            break
    while sum(take.values()) > total:
        for k in sorted(cells, key=lambda k: take[k], reverse=True):
            if sum(take.values()) <= total:
                break
            if take[k] > 1:
                take[k] -= 1
    return take


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-file", default=str(SEG / "sft_export" / "test.jsonl"))
    ap.add_argument("--out-file", default=str(SEG / "sft_export" / "test_strat2000.jsonl"))
    ap.add_argument("--per-procedure", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test_file)]
    rng = random.Random(args.seed)
    picked = []

    for proc in sorted({r["procedure_type"] for r in rows}):
        pool = [r for r in rows if r["procedure_type"] == proc]
        cells = collections.defaultdict(list)
        for r in pool:
            cells[(r["format"], bucket(r["duration"]))].append(r)
        for v in cells.values():
            v.sort(key=lambda r: r["uid"])
            rng.shuffle(v)
        take = allocate(dict(cells), min(args.per_procedure, len(pool)))
        got = [r for k, n in take.items() for r in cells[k][:n]]
        picked += got
        print(f"{proc}: {len(got)} of {len(pool)} rows across {len(cells)} cells")

    picked.sort(key=lambda r: r["uid"])
    with open(args.out_file, "w") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(picked)} rows -> {args.out_file}")


if __name__ == "__main__":
    main()
