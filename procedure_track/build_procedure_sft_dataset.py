"""Builds the chat-formatted SFT dataset (train/eval/test) for the FOCUS PROCEDURE track.

Mirrors `segment_track/build_segment_sft_dataset.py`. Three things differ:

  * the window is a PREFIX `[0, t]` of the whole operation, up to 4 h 56 m, and its end
    is a model input -- 46.5% of repeated (video, question) pairs change their answer
    with it, so `prompts_procedure.window_line()` is prepended to every question;
  * frame indices are stored on the **5 fps grid** the challenge delivers, not at native
    fps (see sampling.py);
  * the frame count is VARIABLE -- short windows come back at true 5 s density using far
    fewer than `--num-frames`, which is the point of the cap.

Usage:
    venv3.12/bin/python procedure_track/build_procedure_sft_dataset.py \
        --datasets heico lapchole --out-dir procedure_track/sft_export
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROC_DIR))
sys.path.insert(0, str(PROC_DIR.parent / "segment_track"))
sys.path.insert(0, str(PROC_DIR.parent / "orena_sft"))

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config  # noqa: E402
from focus.config import DATASET_BASE_FPS  # noqa: E402

from build_frame_sft_dataset import make_eval_video_split  # noqa: E402
from clip_sampling import frame_count, frame_dir  # noqa: E402

from prompts_procedure import window_line  # noqa: E402
from sampling import (  # noqa: E402
    DEFAULT_N_MAX, describe, parse_question_timestamps, sample_indices_5fps, to_native_index,
)

DEFAULT_ROOT_DIR = Path("/projects/datasets_ML/orena/")
DEFAULT_OUT_DIR = PROC_DIR / "sft_export"


def verify_fps(root_dir: Path, dataset: str, video_id: str) -> float | None:
    """The video's real fps.

    Exported 5 fps indices are converted back to native JPEG indices with
    `idx5 * base_fps / 5`. If a video's true rate were 29.97 while the constant says 30,
    every frame lookup would drift -- and the timestamps with it, silently.
    """
    import decord

    path = root_dir / dataset / "videos" / video_id
    if not path.exists():
        return None
    vr = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    return float(vr.get_avg_fps())


def load_records(dataset: str, split: DatasetSplit, root_dir: Path, n_max: int,
                 frames_folder: str) -> list[dict]:
    base_fps = float(DATASET_BASE_FPS[dataset])
    ds = FocusDataset(dataset, split, Track.PROCEDURE)

    records, dropped = [], 0
    for req, ref in ds:
        directory = frame_dir(root_dir, dataset, req.videoID, frames_folder)
        n_avail = frame_count(directory)
        if n_avail == 0:
            dropped += 1
            continue

        # The sampler must see the RAW question: window_line() below contains two
        # timestamps of its own, and parsing after prepending would anchor every row
        # at its own window edges.
        idx5 = sample_indices_5fps(req.start_time, req.end_time, req.question, n_max=n_max)
        idx5 = [i for i in idx5 if to_native_index(i, base_fps) < n_avail]
        if len(idx5) < 2:
            dropped += 1
            continue
        if len(idx5) % 2:
            idx5 = idx5[:-1]

        records.append({
            "uid": f"{dataset}/{req.qID}",
            "qID": req.qID,
            "source_dataset": dataset,
            "videoID": f"{dataset}/{req.videoID}",
            "procedure_type": req.procedure_type,
            "primary_capability": ref.primary.name,
            "secondary_capabilities": [c.name for c in ref.secondaries],
            "format": ref._format,
            "question": req.question,
            "answer": ref.answer,
            "start_time": req.start_time,
            "end_time": req.end_time,
            "duration": req.end_time - req.start_time,
            "base_fps": base_fps,
            "frame_dir": str(directory),
            "frames_indices": idx5,
            "n_frames": len(idx5),
            "n_quoted_timestamps": len(parse_question_timestamps(req.question)),
            **{f"gap_{k}": v for k, v in describe(idx5).items() if k != "n"},
        })
    if dropped:
        print(f"  [{dataset}/{split.value}] dropped {dropped} rows with no usable frames")
    return records


def to_chat_record(r: dict) -> dict:
    """One record for the trainer.

    `messages` holds only the text side; the clip enters as a bare `{"type": "video"}`
    placeholder that the collator fills in from `frame_dir` + `frames_indices`, so
    pixels never touch the JSONL.
    """
    out = {k: r[k] for k in (
        "uid", "qID", "source_dataset", "videoID", "procedure_type", "primary_capability",
        "secondary_capabilities", "format", "start_time", "end_time", "duration",
        "base_fps", "frame_dir", "frames_indices", "n_frames", "n_quoted_timestamps",
        "gap_min_gap_s", "gap_max_gap_s", "gap_at_target",
    )}
    out["messages"] = [
        {"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": f"{window_line(r['end_time'])} {r['question']}"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": str(r["answer"])}]},
    ]
    return out


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(to_chat_record(r)) + "\n")


def summarise(name: str, recs: list[dict]) -> None:
    if not recs:
        return
    n = [r["n_frames"] for r in recs]
    at = [r["gap_at_target"] for r in recs]
    fmt = collections.Counter(r["format"] for r in recs)
    print(f"  {name:6s} {len(recs):6d} rows | frames med {statistics.median(n):.0f} "
          f"max {max(n)} | rows fully at 5s density "
          f"{100 * sum(a >= 0.999 for a in at) / len(at):.1f}% | "
          + " ".join(f"{k}:{v}" for k, v in fmt.most_common()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["heico", "lapchole"],
                    choices=["heico", "lapchole"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--num-frames", type=int, default=DEFAULT_N_MAX,
                    help="UPPER BOUND per row; shorter windows use fewer. Must be even.")
    ap.add_argument("--frames-folder", default="frames_overlay",
                    help="'frames_overlay' (default, burned-in clock) or 'frames'")
    ap.add_argument("--eval-frac", type=float, default=0.10,
                    help="fraction of TRAIN videos (per stratum) held out for eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-verify-fps", action="store_true")
    ap.add_argument("--include-test", action="store_true",
                    help="also write all.jsonl = train + eval + test, for the FINAL "
                         "submission model. The public test split ships with labels and "
                         "the platform scores against its own hidden set, so those rows "
                         "are ordinary training data -- but local evaluation of anything "
                         "trained on it is meaningless. A flag, never a shell pipeline: "
                         "the segment track shipped a model whose training data came "
                         "from an unrecorded `cat`.")
    args = ap.parse_args()

    if args.num_frames % 2:
        ap.error(f"--num-frames must be even, got {args.num_frames}")

    set_config(FocusConfig(root_dir=args.root_dir))

    train_records, test_records = [], []
    for dataset in args.datasets:
        print(f"Loading {dataset!r} procedure track...")
        tr = load_records(dataset, DatasetSplit.TRAIN, args.root_dir,
                          args.num_frames, args.frames_folder)
        te = load_records(dataset, DatasetSplit.TEST, args.root_dir,
                          args.num_frames, args.frames_folder)
        print(f"  train: {len(tr)} rows   test: {len(te)} rows")
        train_records += tr
        test_records += te

    if not args.no_verify_fps:
        print("\nVerifying real fps against DATASET_BASE_FPS...")
        seen, bad = set(), []
        for r in train_records + test_records:
            if r["videoID"] in seen:
                continue
            seen.add(r["videoID"])
            dataset, vid = r["videoID"].split("/", 1)
            actual = verify_fps(args.root_dir, dataset, vid)
            if actual is not None and abs(actual - r["base_fps"]) > 1e-6:
                bad.append((r["videoID"], actual, r["base_fps"]))
        if bad:
            for key, actual, expected in bad:
                print(f"  MISMATCH {key}: real {actual} vs constant {expected}")
            raise SystemExit("fps mismatch would corrupt every frame lookup; aborting.")
        print(f"  OK — {len(seen)} videos match their DATASET_BASE_FPS constant.")

    eval_videos = make_eval_video_split(train_records, args.eval_frac, args.seed)
    eval_records = [r for r in train_records if r["videoID"] in eval_videos]
    final_train = [r for r in train_records if r["videoID"] not in eval_videos]

    print(f"\nVideo-level eval split (eval_frac={args.eval_frac}): {len(eval_videos)} eval "
          f"videos, {len({r['videoID'] for r in final_train})} train videos")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = [("train", final_train), ("eval", eval_records), ("test", test_records)]
    if args.include_test:
        splits.append(("all", final_train + eval_records + test_records))
    for name, recs in splits:
        write_jsonl(recs, args.out_dir / f"{name}.jsonl")
        summarise(name, recs)
    print(f"\nWrote {', '.join(n for n, _ in splits)}.jsonl to {args.out_dir}/")

    s = final_train[0]
    d = describe(s["frames_indices"])
    print(f"\nSanity sample — qID {s['qID']} ({s['format']}), {s['duration']/60:.0f} min "
          f"window, {d['n']} frames, gaps {d['min_gap_s']:.1f}..{d['max_gap_s']:.1f}s, "
          f"{100*d['at_target']:.0f}% at the 5 s target")


if __name__ == "__main__":
    main()
