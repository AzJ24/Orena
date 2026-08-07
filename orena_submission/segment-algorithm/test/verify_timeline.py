"""Checks the container rebuilds the prompt training used, from a platform clip.

Runs on CPU, no weights loaded. Two independent comparisons against every exported
test record:

1. the `<N seconds>` marker times `resources/clip_frames.py` derives from a 5 fps
   trimmed clip vs. the ones `segment_track/clip_sampling.py` derived from the
   source-video JPEGs, and
2. the full prompt text the processor renders from each, byte for byte, sampled
   across both source frame rates.

Training snapped each marker onto the SOURCE video's integer frame grid, which was
25 fps (heico) or 30 fps (lapchole). A request carries no frame rate, so the clip is
re-expressed on a fixed 25 fps grid: exact for 25 fps sources, and off by at most
half a frame (<=20 ms) for 30 fps ones. `focus.data.formats.Time` accepts an answer
within 5 s, so that residual is ~0.6% of the scoring tolerance -- the threshold this
script asserts against.

Usage:
    venv3.12/bin/python orena_submission/segment-algorithm/test/verify_timeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(REPO / "segment_track"))
sys.path.insert(0, str(REPO / "orena_sft"))

from clip_sampling import marker_times  # noqa: E402  (training-side, source JPEGs)
from resources.clip_frames import (  # noqa: E402
    FRAME_SIZE, N_FRAMES, TIMELINE_FPS, build_metadata, sample_indices,
)
from resources.prompts import build_system_prompt  # noqa: E402

CHECKPOINT = REPO / "segment_track/checkpoints/segment-27b-alldata-n80-650-20260730-merged"
TEST_JSONL = REPO / "segment_track/sft_export/test.jsonl"
CLIP_FPS = 5.0  # the platform's clip encoding
DRIFT_TOLERANCE_S = 0.25  # 5% of the 5 s Time acceptance threshold


def n_clip_frames(record: dict) -> int:
    """Frames the platform's 5 fps clip of this window holds (11 s -> 55)."""
    return max(int(round((record["end_time"] - record["start_time"]) * CLIP_FPS)), 1)


def main() -> int:
    records = [json.loads(line) for line in TEST_JSONL.open()]
    print(f"{len(records)} exported test records\n")

    print("[1] marker times vs. training")
    worst = {}
    for r in records:
        train = marker_times(r["frames_indices"], r["base_fps"])
        _, absolute = sample_indices(
            r["start_time"], r["end_time"], CLIP_FPS, n_clip_frames(r), N_FRAMES)
        ours = marker_times(absolute, TIMELINE_FPS)
        assert len(train) == len(ours) == N_FRAMES // 2, "frame count changed"
        drift = max(abs(a - b) for a, b in zip(train, ours))
        worst[r["base_fps"]] = max(worst.get(r["base_fps"], 0.0), drift)

    for fps in sorted(worst):
        n = sum(1 for r in records if r["base_fps"] == fps)
        print(f"    {int(fps)} fps sources ({n:>4} records): worst drift {worst[fps] * 1000:6.1f} ms")
    max_drift = max(worst.values())
    drift_ok = max_drift <= DRIFT_TOLERANCE_S
    print(f"    worst overall: {max_drift * 1000:.1f} ms  "
          f"(tolerance {DRIFT_TOLERANCE_S * 1000:.0f} ms, Time scored at +/-5 s) -> "
          f"{'OK' if drift_ok else 'TOO LARGE'}")

    print("\n[2] rendered prompt text vs. training")
    if not CHECKPOINT.exists():
        print(f"    SKIPPED - checkpoint not found at {CHECKPOINT}")
        return 0 if drift_ok else 1

    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    system_prompt = build_system_prompt(False, style="direct", track="segment")
    # Pixels never reach the rendered text; only the grid shape does.
    dummy = np.zeros((N_FRAMES, FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)

    def render(meta: VideoMetadata, question: str) -> str:
        text = processor.apply_chat_template(
            [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "video"}, {"type": "text", "text": question}]},
            ],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        out = processor(text=[text], videos=[dummy], video_metadata=[meta],
                        do_sample_frames=False, return_tensors="pt")
        return processor.tokenizer.decode(out["input_ids"][0])

    # Sample both frame rates: an all-heico head would hide the 30 fps residual.
    per_fps = {}
    for r in records:
        per_fps.setdefault(r["base_fps"], []).append(r)
    sample = [r for rs in per_fps.values() for r in rs[:15]]

    identical = {}
    for r in sample:
        question = r["messages"][0]["content"][1]["text"]
        train_meta = VideoMetadata(
            total_num_frames=r["frames_indices"][-1] + 1, fps=r["base_fps"],
            width=FRAME_SIZE[0], height=FRAME_SIZE[1],
            duration=(r["frames_indices"][-1] + 1) / r["base_fps"],
            frames_indices=r["frames_indices"])
        _, absolute = sample_indices(
            r["start_time"], r["end_time"], CLIP_FPS, n_clip_frames(r), N_FRAMES)
        our_meta = build_metadata(absolute, FRAME_SIZE)

        same = render(train_meta, question) == render(our_meta, question)
        hit, tot = identical.get(r["base_fps"], (0, 0))
        identical[r["base_fps"]] = (hit + int(same), tot + 1)

    for fps in sorted(identical):
        hit, tot = identical[fps]
        print(f"    {int(fps)} fps sources: {hit}/{tot} prompts byte-identical to training")

    # 25 fps must be exact -- it is the grid the container reconstructs on. 30 fps
    # is allowed to differ, but only in marker digits, which [1] has already bounded.
    exact_ok = identical.get(25.0, (0, 1))[0] == identical.get(25.0, (0, 1))[1]
    print(f"    25 fps exactness -> {'OK' if exact_ok else 'BROKEN'}")

    ok = drift_ok and exact_ok
    print("\n" + ("PASS - prompt reconstruction matches training within scoring tolerance"
                  if ok else "FAIL - reconstruction is off"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
