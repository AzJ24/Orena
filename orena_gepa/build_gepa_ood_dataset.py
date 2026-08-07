"""Build a GEPA dataset for optimizing a prompt for the SFT model, tuned for OOD.

This is the leakage-safe, OOD-oriented variant of `build_gepa_dataset.py`, meant
for running GEPA on a model fine-tuned on the TRAIN split (e.g. the merged
`combined-9b-8r-direct`). Two disjoint pools:

  * train.jsonl -- the reflection pool GEPA samples minibatches from. Drawn from
    the SFT model's HELD-OUT eval videos (the eval split carved from TRAIN, which
    that model never trained on), so reflecting on it measures generalization,
    not memorised training data. The exact held-out videoIDs are read straight
    from `sft_export/combined/eval.jsonl` (ground truth of what SFT held out),
    not re-derived from an RNG split.

  * val.jsonl -- the Pareto/selection signal, drawn from the OOD test condition:
    the heico TEST set, which is entirely the *new* `Sigmoid Resection` procedure
    (training saw only Proctocolectomy / Rectal Resection / Cholecystectomy). So
    GEPA selects the prompt on genuine procedure-level OOD generalization.

Note on OOD FO classes: `Mesh` and `Absorbable Hemostatic Agent` were requested,
but they do not occur as answers anywhere in heico/lapchole (train or test) -- so
the OOD signal here is the new procedure, not novel FO classes. If a future test
phase ships those classes, this script picks them up automatically via the
`--ood-fo-classes` filter below.

Usage:
    .venv/bin/python orena_gepa/build_gepa_ood_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config
from focus.config import DATASET_BASE_FPS
from focus.data.formats import JUDGE_FORMATS

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orena_sft"))
from build_frame_sft_dataset import DEFAULT_ROOT_DIR, frame_path  # noqa: E402
from build_gepa_dataset import stratified_sample  # noqa: E402

GEPA_DIR = Path(__file__).resolve().parent
SFT_DIR = GEPA_DIR.parent / "orena_sft"


def load_records(dataset: str, split: DatasetSplit, cfg: FocusConfig, keep_judge: bool) -> list[dict]:
    """Load frame-track records with everything needed for exact scoring (incl.
    format_kwargs) plus procedure_type for OOD filtering."""
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
            "start_time": req.start_time,
            "end_time": req.end_time,
        })
    return records


def sft_heldout_video_ids(eval_jsonl: Path) -> set[str]:
    """The exact set of (qualified) videoIDs the SFT model held out for eval."""
    vids = set()
    with eval_jsonl.open() as f:
        for line in f:
            vids.add(json.loads(line)["videoID"])
    return vids


def answer_has_fo_class(record: dict, classes: list[str]) -> bool:
    if record["format"] != "fo_class":
        return False
    ans = {a.strip().lower() for a in record["answer"].split(",")}
    return any(c.lower() in ans for c in classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--out-dir", type=Path, default=GEPA_DIR / "gepa_data_ood")
    ap.add_argument("--sft-eval-jsonl", type=Path,
                     default=SFT_DIR / "sft_export" / "combined" / "eval.jsonl",
                     help="the SFT model's held-out eval split; its videoIDs source the GEPA train pool")
    ap.add_argument("--train-datasets", nargs="+", default=["heico", "lapchole"],
                     help="datasets to draw the SFT-held-out reflection pool from")
    ap.add_argument("--ood-dataset", default="heico", choices=["heico", "lapchole"],
                     help="dataset whose TEST split provides the OOD val (heico test = Sigmoid Resection)")
    ap.add_argument("--ood-procedures", nargs="*", default=["Sigmoid Resection"],
                     help="restrict the OOD val to these procedure_type(s); empty = all test procedures")
    ap.add_argument("--ood-fo-classes", nargs="*", default=["Mesh", "Absorbable Hemostatic Agent"],
                     help="if any of these FO classes appear in the test split, ADD those questions to "
                          "the OOD val (currently absent in heico/lapchole; picked up automatically if a "
                          "future test phase ships them)")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=120)
    ap.add_argument("--include-judge-formats", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    # ── GEPA train pool: SFT-held-out eval videos ────────────────────────────
    heldout = sft_heldout_video_ids(args.sft_eval_jsonl)
    print(f"SFT held-out videos: {len(heldout)}")
    train_pool = []
    for ds in args.train_datasets:
        recs = load_records(ds, DatasetSplit.TRAIN, cfg, args.include_judge_formats)
        kept = [r for r in recs if r["videoID"] in heldout]
        print(f"  {ds} TRAIN: {len(kept)}/{len(recs)} rows fall in SFT-held-out videos")
        train_pool += kept
    train = stratified_sample(train_pool, args.n_train, args.seed, exclude_qids=set())

    # ── GEPA OOD val: new-procedure test set (+ any OOD FO-class questions) ───
    test_recs = load_records(args.ood_dataset, DatasetSplit.TEST, cfg, args.include_judge_formats)
    proc_set = set(args.ood_procedures)
    ood_pool = [r for r in test_recs if (not proc_set or r["procedure_type"] in proc_set)]
    fo_hits = [r for r in test_recs if answer_has_fo_class(r, args.ood_fo_classes)]
    if fo_hits:
        seen = {r["qID"] for r in ood_pool}
        ood_pool += [r for r in fo_hits if r["qID"] not in seen]
    print(f"  {args.ood_dataset} TEST OOD pool: {len(ood_pool)} rows "
          f"(procedures={args.ood_procedures}, fo-class hits={len(fo_hits)})")
    val = stratified_sample(ood_pool, args.n_val, args.seed, exclude_qids=set())

    # ── report + write ───────────────────────────────────────────────────────
    def mix(rows, key):
        c = {}
        for r in rows:
            c[r[key]] = c.get(r[key], 0) + 1
        return dict(sorted(c.items(), key=lambda kv: -kv[1]))

    print(f"\nGEPA train (SFT-held-out, in-distribution procedures): {len(train)}")
    print(f"  procedures: {mix(train, 'procedure_type')}")
    print(f"  formats:    {mix(train, 'format')}")
    print(f"  capability: {mix(train, 'primary_capability')}")
    print(f"GEPA OOD val (new procedure): {len(val)}")
    print(f"  procedures: {mix(val, 'procedure_type')}")
    print(f"  formats:    {mix(val, 'format')}")
    print(f"  capability: {mix(val, 'primary_capability')}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
