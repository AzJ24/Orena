"""Round-trip check: does a 5 fps clip reproduce the absolute timestamps training used?

Training sampled JPEGs at native fps and passed `VideoMetadata(fps=base_fps,
frames_indices=<native indices>)`, so the processor rendered ABSOLUTE video time.
At submission the clip arrives as an H.264 MP4 at exactly 5 fps, trimmed to the
window, and must produce the SAME marker values or every `time` answer is wrong.

Encodes a real training row to 5 fps, decodes it back, runs the real processor, and
compares the rendered markers against the training path. Also exercises the two ways
of getting it wrong, because both fail silently.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import decord
import numpy as np

SEG = Path(__file__).resolve().parents[1] / "segment_track"
sys.path.insert(0, str(SEG))

from clip_sampling import frame_file, marker_times  # noqa: E402

MARKER_RE = r"<(\d+\.\d) seconds><\|vision_start\|>"
ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


def encode_5fps(frame_dir, start, end, base_fps, out_path, max_height=576, workers=32):
    """The challenge's encoder, as documented: exactly 5 fps, height <= 576, trimmed.

    Reads through a thread pool: this NFS path is latency-bound at ~71 ms/frame, so a
    5-minute clip is minutes single-threaded and seconds in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = int((end - start) * 5) + 1
    native = [round((start + k / 5.0) * base_fps) for k in range(n)]
    with ThreadPoolExecutor(workers) as pool:
        imgs = list(pool.map(lambda i: cv2.imread(str(frame_file(frame_dir, i))), native))
    h, w = imgs[0].shape[:2]
    if h > max_height:
        w, h = round(w * max_height / h), max_height
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (w, h))
    for img in imgs:
        writer.write(cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img)
    writer.release()
    return n, (w, h)


def render_markers(processor, video, meta):
    out = processor(text=["<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>ok<|im_end|>\n"],
                    videos=[video], video_metadata=[meta],
                    do_sample_frames=False, return_tensors="pt")
    text = processor.tokenizer.decode(out["input_ids"][0])
    return [float(x) for x in re.findall(MARKER_RE, text)]


def main():
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    tmp = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fpscheck")
    tmp.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")

    # Two windows: one starting mid-video (the SEGMENT case, where the index origin
    # matters) and one starting at zero (every PROCEDURE row).
    cases = [
        ("SEGMENT-like (start != 0)", "heico", "0009 - Heico - Prokto - 10", 25.0, 555.0, 854.0, 16),
        ("PROCEDURE-like (start == 0)", "heico", "0009 - Heico - Prokto - 10", 25.0, 0.0, 600.0, 16),
    ]

    for label, ds, stem, base_fps, start, end, n_sample in cases:
        print(f"\n=== {label}: [{start:.0f}s, {end:.0f}s] @ {base_fps} fps source")
        fdir = Path(f"/projects/datasets_ML/orena/{ds}/frames/{stem}")
        mp4 = tmp / f"{stem[:12].strip().replace(' ', '_')}_{int(start)}.mp4"

        n_clip, (w, h) = encode_5fps(fdir, start, end, base_fps, mp4)
        print(f"  encoded {n_clip} frames @ 5 fps, {w}x{h}, {mp4.stat().st_size/1e6:.1f} MB")

        vr = decord.VideoReader(str(mp4), ctx=decord.cpu(0), num_threads=1)
        check("decoded frame count matches", len(vr) == n_clip, f"{len(vr)} vs {n_clip}")
        check("decoded fps is 5.0", abs(vr.get_avg_fps() - 5.0) < 1e-6, f"{vr.get_avg_fps()}")

        local = [int(round(x)) for x in np.linspace(0, n_clip - 1, n_sample)]
        video = vr.get_batch(local).asnumpy()

        # --- the training path, for reference -------------------------------
        native = [round((start + k / 5.0) * base_fps) for k in local]
        train_expected = marker_times(native, base_fps)

        # --- the CORRECT inference path -------------------------------------
        idx5 = [round(start * 5) + k for k in local]
        meta = VideoMetadata(total_num_frames=n_clip, fps=5.0, width=w, height=h,
                             duration=n_clip / 5.0, frames_indices=idx5)
        got = render_markers(processor, video, meta)
        check(f"{n_sample//2} markers rendered", len(got) == n_sample // 2, str(len(got)))
        check("markers match the TRAINING path",
              all(abs(a - b) < 0.05 for a, b in zip(got, train_expected)),
              f"first {got[0]} vs {train_expected[0]:.1f}")
        # The first marker is the MEAN of the first fused pair, so it sits half a
        # sampling interval past the window start -- not at it.
        interval = (end - start) / (n_sample - 1)
        check("markers are absolute video time, not clip-relative",
              start - 0.05 <= got[0] <= start + interval + 0.05,
              f"{got[0]}s in [{start}, {start + interval:.1f}]s (pair mean, "
              f"interval {interval:.1f}s)")

        # --- failure mode 1: keep the training fps --------------------------
        bad = VideoMetadata(total_num_frames=n_clip, fps=base_fps, width=w, height=h,
                            duration=n_clip / base_fps, frames_indices=idx5)
        got_bad = render_markers(processor, video, bad)
        check("wrong fps IS caught by this test", abs(got_bad[0] - got[0]) > 1.0,
              f"fps={base_fps} gives {got_bad[0]}s instead of {got[0]}s "
              f"({got[0]/max(got_bad[0],1e-9):.1f}x off)")

        # --- failure mode 2: forget the origin shift ------------------------
        bad2 = VideoMetadata(total_num_frames=n_clip, fps=5.0, width=w, height=h,
                             duration=n_clip / 5.0, frames_indices=local)
        got_bad2 = render_markers(processor, video, bad2)
        drift = abs(got_bad2[0] - got[0])
        check("unshifted origin IS caught by this test" if start else
              "unshifted origin is HARMLESS when start == 0",
              drift > 1.0 if start else drift < 0.05,
              f"gives {got_bad2[0]}s vs correct {got[0]}s (drift {drift:.1f}s)")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
