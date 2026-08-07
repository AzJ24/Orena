"""Build a realistic /input batch from the real FOCUS TEST split, in the platform's
exact layout — so the container can be timed and scored on real questions instead of
the template's three placeholder clips.

The clips are re-encoded to the platform's stated format rather than approximated:
H.264 MP4, **exactly 5 fps**, height normalised to at most 576 px with the width
following the source aspect ratio, and a keyframe every 5 s. Source pixels come from
the same extracted JPEGs training read, so the only difference from training's view
is the one the platform imposes — the 5 fps resampling and the trim to the window.

Ground truth is written alongside (`references.json`) but OUTSIDE the /input tree,
since the platform does not provide it.

Usage:
    venv3.12/bin/python orena_submission/segment-algorithm/test/build_real_batch.py \
        --n-per-dataset 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "segment_track"))

from clip_sampling import frame_count, frame_file  # noqa: E402
from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config  # noqa: E402
from focus.config import DATASET_BASE_FPS  # noqa: E402
from focus.foreign_objects import FO_DEFINITIONS_FILE  # noqa: E402

DEFAULT_ROOT = Path("/projects/datasets_ML/orena/")
CLIP_FPS = 5.0        # the platform's clip frame rate
MAX_HEIGHT = 576      # the platform's height cap
KEYFRAME_S = 5        # "a keyframe every 5 s"


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def target_size(width: int, height: int) -> tuple[int, int]:
    """Height capped at 576, width by source aspect, both even (H.264 yuv420p)."""
    if height <= MAX_HEIGHT:
        w, h = width, height
    else:
        h = MAX_HEIGHT
        w = int(round(width * MAX_HEIGHT / height))
    return w - (w % 2), h - (h % 2)


def encode_clip(frames: list[Path], out_path: Path, size: tuple[int, int]) -> None:
    """Pipe the sampled JPEGs into ffmpeg as a 5 fps H.264 clip."""
    w, h = size
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(CLIP_FPS), "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-g", str(int(CLIP_FPS * KEYFRAME_S)),
        "-r", str(CLIP_FPS),
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for p in frames:
        img = Image.open(p).convert("RGB")
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        proc.stdin.write(np.asarray(img).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "real_batch")
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--n-per-dataset", type=int, default=10)
    ap.add_argument("--max-duration", type=float, default=120.0,
                    help="skip longer windows so the batch encodes quickly")
    args = ap.parse_args()

    set_config(FocusConfig(root_dir=args.root_dir))

    interface = args.out / "interface_1"
    if interface.exists():
        shutil.rmtree(interface)
    (interface / "plain").mkdir(parents=True)

    # The export carries the frame_dir/base_fps mapping FocusDataset does not.
    by_qid = {}
    with (REPO / "segment_track/sft_export/test.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            by_qid[r["qID"]] = r

    requests, references = [], []
    n = 0
    for dataset in ("heico", "lapchole"):
        base_fps = float(DATASET_BASE_FPS[dataset])
        ds = FocusDataset(dataset, DatasetSplit.TEST, Track.SEGMENT)

        # Round-robin over answer formats rather than taking the first N. Taking
        # them in dataset order yields 20 fo_class questions and never exercises
        # `time` -- which is 38.5% of segment data and the ONLY bucket that tests
        # the absolute-timestamp reconstruction, i.e. the riskiest code here.
        buckets: dict[str, list[int]] = {}
        for i in range(len(ds)):
            _req, _ref = ds[i]
            buckets.setdefault(_ref._format, []).append(i)
        order: list[int] = []
        for rank in range(max((len(v) for v in buckets.values()), default=0)):
            for fmt in sorted(buckets):
                if rank < len(buckets[fmt]):
                    order.append(buckets[fmt][rank])

        taken = 0
        for i in order:
            if taken >= args.n_per_dataset:
                break
            req, ref = ds[i]
            rec = by_qid.get(req.qID)
            if rec is None or req.duration > args.max_duration or req.duration < 1:
                continue
            directory = Path(rec["frame_dir"])
            n_src = frame_count(directory)
            if not n_src:
                continue

            # The clip the platform would cut: 5 fps across [start, end].
            n_clip = max(int(round(req.duration * CLIP_FPS)), 1)
            src_idx = [min(int(round((req.start_time + k / CLIP_FPS) * base_fps)), n_src - 1)
                       for k in range(n_clip)]
            paths = [frame_file(directory, j) for j in src_idx]
            if not all(p.exists() for p in paths):
                continue

            n += 1
            qid = f"q{n:03d}"
            size = target_size(*Image.open(paths[0]).size)
            encode_clip(paths, interface / "plain" / f"{qid}.mp4", size)

            requests.append({
                "qID": qid, "videoID": req.videoID,
                "start_time": float(req.start_time), "end_time": float(req.end_time),
                "procedure_type": req.procedure_type, "question": req.question,
            })
            references.append({
                "qID": qid, "orig_qID": req.qID, "dataset": dataset,
                "format": ref._format, "answer": ref.answer,
                "primary": ref.primary.name, "duration": float(req.duration),
                "format_kwargs": ref.format_kwargs,
            })
            taken += 1
            print(f"  {qid}  {dataset:8s}  {req.duration:6.1f}s  {n_clip:4d} frames "
                  f"{size[0]}x{size[1]}  [{ref._format}]")
        print(f"{dataset}: {taken} questions")

    (interface / "request.json").write_text(json.dumps(requests, indent=2))
    (interface / "FO_definitions.json").write_text(
        json.dumps(FO_DEFINITIONS_FILE.read_text()))
    (interface / "batch.json").write_text(json.dumps({
        "qIDs": [r["qID"] for r in requests], "batch_size": len(requests),
        "layout": {"plain": "plain/<qID>.mp4"},
    }, indent=2))
    # Ground truth stays out of /input — the platform never provides it.
    (args.out / "references.json").write_text(json.dumps(references, indent=2))

    total_mb = sum(p.stat().st_size for p in (interface / "plain").glob("*.mp4")) / 1e6
    print(f"\n{len(requests)} questions -> {interface}  ({total_mb:.1f} MB of clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
