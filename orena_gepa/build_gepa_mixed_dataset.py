"""Build a MIXED ID+OOD GEPA dataset for optimizing a prompt for the SFT model.

A refinement of `build_gepa_ood_dataset.py` that addresses the two weaknesses seen
when a prompt optimized on OOD-only / deterministic-only data was tested:
  * it regressed the judge formats (multiple_choice / open_ended) -- they were
    never in the optimization;
  * it traded away in-distribution performance -- the val was OOD-only, but the
    challenge metric weights ID and OOD equally.

So both splits here mix ID and OOD, include judge formats, and are split at the
VIDEO level so train (reflection) and val (selection) never share a surgery.

Sources -- all held out from the TRAIN-only model this optimizes for:
  * OOD: heico TEST (Sigmoid Resection, a procedure unseen in training).
  * ID : the SFT model's held-out eval videos (Proctocolectomy / Rectal /
         Cholecystectomy) + lapchole TEST (Cholecystectomy) -- seen procedures,
         held-out videos.

This does NOT touch build_gepa_ood_dataset.py, gepa_data_ood, or any prior run.

Usage:
    .venv/bin/python orena_gepa/build_gepa_mixed_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from focus import DatasetSplit, FocusConfig, set_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orena_sft"))
from build_frame_sft_dataset import DEFAULT_ROOT_DIR  # noqa: E402
from build_gepa_dataset import stratified_sample  # noqa: E402
from build_gepa_ood_dataset import load_records, sft_heldout_video_ids  # noqa: E402

GEPA_DIR = Path(__file__).resolve().parent
SFT_DIR = GEPA_DIR.parent / "orena_sft"


def split_by_video(records: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Partition records into (train, val) with DISJOINT videoIDs, so no surgery
    appears in both the reflection and selection sets."""
    rng = np.random.RandomState(seed)
    videos = sorted({r["videoID"] for r in records})
    rng.shuffle(videos)
    n_val = max(1, round(len(videos) * val_frac))
    val_videos = set(videos[:n_val])
    train = [r for r in records if r["videoID"] not in val_videos]
    val = [r for r in records if r["videoID"] in val_videos]
    return train, val


def tag(records: list[dict], dist: str) -> list[dict]:
    for r in records:
        r["dist"] = dist
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--out-dir", type=Path, default=GEPA_DIR / "gepa_data_mixed")
    ap.add_argument("--sft-eval-jsonl", type=Path,
                     default=SFT_DIR / "sft_export" / "combined" / "eval.jsonl")
    ap.add_argument("--id-train-datasets", nargs="+", default=["heico", "lapchole"],
                     help="datasets whose SFT-held-out TRAIN videos seed the ID pool")
    ap.add_argument("--id-test-dataset", default="lapchole",
                     help="TEST split of a SEEN procedure to add to the ID pool (lapchole=Cholecystectomy)")
    ap.add_argument("--ood-dataset", default="heico",
                     help="TEST split providing OOD (heico test = Sigmoid Resection)")
    ap.add_argument("--ood-procedures", nargs="*", default=["Sigmoid Resection"])
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=180)
    ap.add_argument("--val-ood-frac", type=float, default=0.5,
                     help="fraction of the VAL that is OOD (0.5 = balanced, matches the equal-weight metric)")
    ap.add_argument("--train-ood-frac", type=float, default=0.6,
                     help="fraction of the TRAIN (reflection) pool that is OOD")
    ap.add_argument("--video-val-frac", type=float, default=0.5,
                     help="fraction of each pool's VIDEOS reserved for val (rest -> train); keeps them disjoint")
    ap.add_argument("--no-judge-formats", action="store_true",
                     help="exclude open_ended/matching/multiple_choice (default: include them)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keep_judge = not args.no_judge_formats
    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    # ── build the two source pools (all held out from the TRAIN-only model) ──
    heldout = sft_heldout_video_ids(args.sft_eval_jsonl)
    id_pool: list[dict] = []
    for ds in args.id_train_datasets:
        recs = load_records(ds, DatasetSplit.TRAIN, cfg, keep_judge)
        id_pool += [r for r in recs if r["videoID"] in heldout]
    id_pool += load_records(args.id_test_dataset, DatasetSplit.TEST, cfg, keep_judge)

    proc_set = set(args.ood_procedures)
    ood_all = load_records(args.ood_dataset, DatasetSplit.TEST, cfg, keep_judge)
    ood_pool = [r for r in ood_all if (not proc_set or r["procedure_type"] in proc_set)]

    print(f"ID pool: {len(id_pool)} rows ({len({r['videoID'] for r in id_pool})} videos)")
    print(f"OOD pool: {len(ood_pool)} rows ({len({r['videoID'] for r in ood_pool})} videos)")

    # ── video-disjoint train/val split within each pool ──────────────────────
    id_tr, id_val = split_by_video(id_pool, args.video_val_frac, args.seed)
    ood_tr, ood_val = split_by_video(ood_pool, args.video_val_frac, args.seed + 1)

    # ── compose val (balanced) and train (OOD-leaning), stratified within each ─
    n_val_ood = round(args.n_val * args.val_ood_frac)
    n_val_id = args.n_val - n_val_ood
    val = (tag(stratified_sample(ood_val, n_val_ood, args.seed, set()), "OOD")
           + tag(stratified_sample(id_val, n_val_id, args.seed, set()), "ID"))

    n_train_ood = round(args.n_train * args.train_ood_frac)
    n_train_id = args.n_train - n_train_ood
    train = (tag(stratified_sample(ood_tr, n_train_ood, args.seed + 2, set()), "OOD")
             + tag(stratified_sample(id_tr, n_train_id, args.seed + 2, set()), "ID"))

    # ── report ───────────────────────────────────────────────────────────────
    def mix(rows, key):
        c = {}
        for r in rows:
            c[r[key]] = c.get(r[key], 0) + 1
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))

    for name, rows in (("TRAIN (reflection)", train), ("VAL (selection)", val)):
        print(f"\n{name}: {len(rows)}")
        print(f"  dist:       {mix(rows, 'dist')}")
        print(f"  procedures: {mix(rows, 'procedure_type')}")
        print(f"  formats:    {mix(rows, 'format')}")
        print(f"  capability: {mix(rows, 'primary_capability')}")

    # sanity: no shared video between train and val
    shared = {r["videoID"] for r in train} & {r["videoID"] for r in val}
    print(f"\nshared videos between train & val: {len(shared)} (must be 0)")
    assert not shared, shared

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
