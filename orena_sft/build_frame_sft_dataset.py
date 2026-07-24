"""Builds a chat-formatted SFT dataset (train/eval/test) for VLM fine-tuning
on the FOCUS frame track.

Unlike `build_probing_dataset.py`, this script does NOT read from the
`webapp/cache/qa_cache.parquet` cache — it loads the QA pairs itself, straight
from HuggingFace (or its local cache, same as `data_exploration.ipynb`), via
`focus.FocusDataset`. Only the frame track is used (single still image per
question, `duration == 0`).

Images are never copied or embedded: each record just references the
already-extracted frame JPEG on disk (under `<root_dir>/<dataset>/frames/`),
resolved with the exact same formula `FocusFrameDataset` uses internally
(`round(start_time * base_fps)`). This keeps the output at KB scale instead of
duplicating gigabytes of pixel data that's already sitting on local disk.

`eval` is carved out of `train` at the *video* level (never at the question
level, since many frame-track questions from the same video are correlated),
stratified by procedure_type so every split keeps all procedure types
represented. The official `test` split from HuggingFace is left untouched, as
the final held-out benchmark.

Pass multiple `--datasets` to merge them into one combined export (e.g. both
heico and lapchole) -- videoIDs are qualified per-dataset so the eval split
never treats same-named videos from different datasets as one video, and
each record keeps a `source_dataset` field for later filtering/analysis.

Usage:
    .venv/bin/python orena_sft/build_frame_sft_dataset.py --datasets heico lapchole
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config
from focus.config import DATASET_BASE_FPS

DEFAULT_ROOT_DIR = Path("/projects/datasets_ML/orena/")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "sft_export"


# ── frame path resolution (matches FocusFrameDataset internals) ────────────

def frame_path(cfg: FocusConfig, dataset: str, base_fps: float, video_id: str, t: float) -> Path:
    video_stem = Path(video_id).stem
    frame_idx = round(t * base_fps)
    return cfg.get_dataset_dir(dataset) / cfg.FRAMES_FOLDER / video_stem / f"frame{frame_idx:07d}.jpg"


# ── loading QA data directly (not from cache) ───────────────────────────────

def load_frame_track_records(dataset: str, split: DatasetSplit, cfg: FocusConfig) -> list[dict]:
    base_fps = float(DATASET_BASE_FPS[dataset])
    ds = FocusDataset(dataset, split, Track.FRAME)

    records = []
    dropped = 0
    for req, ref in ds:
        p = frame_path(cfg, dataset, base_fps, req.videoID, req.start_time)
        if not p.exists():
            dropped += 1
            continue
        records.append({
            "qID": req.qID,
            "source_dataset": dataset,
            # videoID is only unique within one dataset; qualify it so a
            # video-level split never accidentally merges same-named videos
            # from different datasets into one bucket.
            "videoID": f"{dataset}/{req.videoID}",
            "procedure_type": req.procedure_type,
            "primary_capability": ref.primary.name,
            "secondary_capabilities": [c.name for c in ref.secondaries],
            "format": ref._format,
            "question": req.question,
            "answer": ref.answer,
            "image_path": str(p),
        })
    if dropped:
        print(f"  [{dataset}/{split.value}] dropped {dropped} rows with no frame on disk "
              f"(run FrameExtractorPreprocessor first if this is unexpectedly high).")
    return records


# ── video-level eval split, stratified by procedure_type ───────────────────

def make_eval_video_split(records: list[dict], eval_frac: float, seed: int) -> set[str]:
    """Return the set of videoIDs from `records` held out for eval."""
    rng = np.random.RandomState(seed)
    video_to_proc: dict[str, str] = {}
    for r in records:
        video_to_proc.setdefault(r["videoID"], r["procedure_type"])

    proc_to_videos: dict[str, list[str]] = {}
    for v, proc in video_to_proc.items():
        proc_to_videos.setdefault(proc, []).append(v)

    eval_videos: set[str] = set()
    for proc, videos in proc_to_videos.items():
        videos = sorted(videos)  # deterministic order before shuffling
        rng.shuffle(videos)
        n_eval = max(1, round(len(videos) * eval_frac))
        eval_videos.update(videos[:n_eval])
    return eval_videos


# ── chat-format serialization ───────────────────────────────────────────────

def to_chat_record(r: dict) -> dict:
    return {
        "qID": r["qID"],
        "source_dataset": r["source_dataset"],
        "videoID": r["videoID"],
        "procedure_type": r["procedure_type"],
        "primary_capability": r["primary_capability"],
        "secondary_capabilities": r["secondary_capabilities"],
        "format": r["format"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": r["image_path"]},
                    {"type": "text", "text": r["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(r["answer"])}],
            },
        ],
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(to_chat_record(r)) + "\n")


# ── orchestration ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["heico"], choices=["heico", "lapchole"],
                     help="one or more FOCUS datasets to load and merge, e.g. --datasets heico lapchole")
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR,
                     help="root dir containing <dataset>/frames/ for each dataset (passed to FocusConfig)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--eval-frac", type=float, default=0.12,
                     help="fraction of TRAIN videos (per dataset, per procedure_type) held out for eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    train_records, test_records = [], []
    for dataset in args.datasets:
        print(f"Loading {dataset!r} frame track from HuggingFace (train + test)...")
        ds_train = load_frame_track_records(dataset, DatasetSplit.TRAIN, cfg)
        ds_test = load_frame_track_records(dataset, DatasetSplit.TEST, cfg)
        print(f"  train: {len(ds_train)} usable rows")
        print(f"  test:  {len(ds_test)} usable rows")
        train_records += ds_train
        test_records += ds_test

    eval_videos = make_eval_video_split(train_records, args.eval_frac, args.seed)
    eval_records = [r for r in train_records if r["videoID"] in eval_videos]
    final_train_records = [r for r in train_records if r["videoID"] not in eval_videos]

    n_train_videos = len({r["videoID"] for r in final_train_records})
    n_eval_videos = len(eval_videos)
    print(f"\nVideo-level eval split: {n_eval_videos} eval videos, {n_train_videos} train videos "
          f"(target eval_frac={args.eval_frac}).")
    print(f"  final train: {len(final_train_records)} rows")
    print(f"  eval:        {len(eval_records)} rows")
    print(f"  test:        {len(test_records)} rows (untouched HF test split)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(final_train_records, args.out_dir / "train.jsonl")
    write_jsonl(eval_records, args.out_dir / "eval.jsonl")
    write_jsonl(test_records, args.out_dir / "test.jsonl")
    print(f"\nWrote train.jsonl, eval.jsonl, test.jsonl to {args.out_dir}/")


if __name__ == "__main__":
    main()
