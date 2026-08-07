"""Checks clip_sampling against a real segment-track row and real frames on disk."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from clip_sampling import (  # noqa: E402
    build_metadata, frame_count, frame_dir, frame_file, hhmmss,
    load_frames, marker_times, sample_frame_indices, video_stem,
)

ROOT = Path("/projects/datasets_ML/orena")
# Real heico segment row (id 17229): 299 s clip, answer 00:10:12.
VIDEO = "0009 - Heico - Prokto - 10.avi"
FPS = 25.0
START, END = 9 * 60 + 15, 14 * 60 + 14      # 00:09:15 -> 00:14:14
ANSWER = 10 * 60 + 12                        # 00:10:12

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


print("1. paths")
d = frame_dir(ROOT, "heico", VIDEO)
check("stem strips extension", video_stem(VIDEO) == "0009 - Heico - Prokto - 10")
check("frame dir exists", d.is_dir(), str(d))
check("index -> filename", frame_file(d, 13875).name == "frame0013875.jpg")

print("2. frame_count (binary search vs disk)")
n_avail = frame_count(d)
check("plausible count", n_avail > 300_000, f"{n_avail} frames")
check("last frame exists", frame_file(d, n_avail - 1).exists())
check("one past last is absent", not frame_file(d, n_avail).exists())

print("3. sampling policy")
idx = sample_frame_indices(START, END, FPS, n_frames=64, n_available=n_avail)
check("exactly 64", len(idx) == 64)
check("even count", len(idx) % 2 == 0)
check("monotone non-decreasing", all(b >= a for a, b in zip(idx, idx[1:])))
check("starts at clip start", idx[0] == round(START * FPS), f"{idx[0]} vs {round(START*FPS)}")
check("ends at clip end", idx[-1] == round(END * FPS), f"{idx[-1]} vs {round(END*FPS)}")
check("within available frames", idx[-1] < n_avail)

try:
    sample_frame_indices(START, END, FPS, n_frames=63)
    check("odd n_frames rejected", False, "no error raised")
except ValueError:
    check("odd n_frames rejected", True)

print("4. short clip -> repeated indices, still 64")
short = sample_frame_indices(100.0, 101.0, FPS, n_frames=64)
check("still 64 entries", len(short) == 64)
check("has duplicates", len(set(short)) < 64, f"{len(set(short))} distinct")
check("spans the 1 s window", short[0] == 2500 and short[-1] == 2525)

print("5. timestamps are absolute video time")
times = marker_times(idx, FPS)
check("32 markers for 64 frames", len(times) == 32)
check("first marker ~clip start", abs(times[0] - START) < 5, f"{times[0]:.1f}s = {hhmmss(times[0])}")
check("last marker ~clip end", abs(times[-1] - END) < 5, f"{times[-1]:.1f}s = {hhmmss(times[-1])}")
below = [t for t in times if t <= ANSWER]
above = [t for t in times if t > ANSWER]
gap = above[0] - below[-1]
check("answer is bracketed", below and above,
      f"{hhmmss(below[-1])} < 00:10:12 < {hhmmss(above[0])}, gap {gap:.1f}s")
check("gap matches N=64 prediction (~9.6 s)", 8.0 < gap < 11.0, f"{gap:.2f}s")

print("6. metadata")
meta = build_metadata(idx, FPS, n_avail)
check("fps preserved exactly", meta.fps == FPS)
check("frames_indices absolute", meta.frames_indices[0] == idx[0] > 10_000)
check("total_num_frames is source length", meta.total_num_frames == n_avail)
try:
    build_metadata(idx, 0, n_avail)
    check("zero fps rejected", False, "no error raised")
except ValueError:
    check("zero fps rejected", True)

print("7. pixels (loads 8 frames only, to keep the test quick)")
paths = [frame_file(d, i) for i in idx[:8]]
video = load_frames(paths)
check("shape (T,H,W,3)", video.shape == (8, 360, 640, 3), str(video.shape))
check("uint8", video.dtype == np.uint8)
check("not blank", video.std() > 5, f"std {video.std():.1f}")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
