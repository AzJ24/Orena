"""Challenge clip -> the exact input `segment_track/clip_sampling.py` produced.

Training never decoded a video: `clip_sampling.py` read pre-extracted JPEGs of the
SOURCE procedure video, took `N_FRAMES` indices spanning `[start_time, end_time]`,
resized them to 640x360, and handed the processor a `VideoMetadata` carrying the
video's real fps and ABSOLUTE frame indices. Those absolute indices are what make
Qwen3-VL render `<1234.5 seconds>` markers on the SOURCE timeline -- and the system
prompt tells the model to read `time` answers straight off them (38.5% of segment
questions). Reproducing that timeline is therefore the whole job of this module.

The platform instead hands us an H.264 clip already trimmed to the window, at 5 fps.
Two halves have to be rebuilt from it:

* **Marker times** -- `Qwen3VLProcessor._calculate_timestamps` renders one marker per
  fused frame pair as `frames_indices[i] / fps`, and uses nothing else from the
  metadata (`total_num_frames` is read only on the `do_sample_frames=True` path).
  So any `(indices, fps)` pair whose quotients are the wanted absolute seconds
  reproduces training's markers exactly; we take the times from
  `linspace(start_time, end_time, N_FRAMES)` -- what training sampled -- and express
  them against `TIMELINE_FPS`.
* **Pixels** -- the nearest frame the 5 fps clip actually holds for each of those
  times. Clips preserve the source aspect ratio, so resizing to 640x360 lands on the
  same geometry training's source frames did.

Below ~16 s the clip holds fewer than `N_FRAMES` distinct frames and neighbouring
picks repeat (4.1% of test rows; training saw the same effect, just at 25-30 fps
rather than 5). Repeating rather than shortening keeps the sequence length -- and so
the memory ceiling -- constant, exactly as training did.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from transformers.video_utils import VideoMetadata

N_FRAMES = 80  # must stay even: temporal_patch_size=2 fuses frames in pairs
FRAME_SIZE = (640, 360)  # (width, height), as trained

# Training snapped every marker onto the SOURCE video's integer frame grid: 25 fps
# for heico, 30 fps for lapchole. A Request carries no frame rate, so one grid has
# to be assumed. 25 reproduces the heico prompt byte-for-byte and leaves lapchole
# markers off by at most half a frame (<=20 ms, 30 ms measured worst case) -- against
# the 5 s acceptance threshold of `focus.data.formats.Time`, i.e. 0.6% of tolerance.
# test/verify_timeline.py asserts both halves of that claim.
TIMELINE_FPS = 25.0


def sample_indices(start_time: float, end_time: float, clip_fps: float, n_clip_frames: int,
                   n_frames: int = N_FRAMES) -> tuple[list[int], list[int]]:
    """`(clip_indices, absolute_indices)` for one request window.

    `clip_indices` index the trimmed clip; `absolute_indices` are the source-video
    indices the same instants sit at, and are what the processor turns into markers.
    """
    if n_frames % 2:
        raise ValueError(f"n_frames must be even (temporal_patch_size=2), got {n_frames}")
    if end_time < start_time:
        raise ValueError(f"end_time {end_time} precedes start_time {start_time}")
    if n_clip_frames < 1:
        raise ValueError("clip holds no frames")

    times = np.linspace(start_time, end_time, n_frames)
    absolute = [int(round(t * TIMELINE_FPS)) for t in times]
    # The clip starts at start_time, so offsets into it are relative seconds. The
    # last frame of an 11 s / 5 fps clip sits at 10.8 s, so end_time itself clamps
    # onto it rather than running off the end.
    clip = [int(np.clip(round((t - start_time) * clip_fps), 0, n_clip_frames - 1)) for t in times]
    return clip, absolute


def build_metadata(absolute_indices: list[int], size: tuple[int, int] = FRAME_SIZE) -> VideoMetadata:
    """Metadata that makes the processor emit ABSOLUTE source-video timestamps.

    `total_num_frames` never reaches the rendered prompt -- the video processor reads
    it only when it does its own sampling, which `do_sample_frames=False` disables --
    but it is a required field, so it gets the source length implied by the last
    sampled instant.
    """
    width, height = size
    return VideoMetadata(
        total_num_frames=absolute_indices[-1] + 1,
        fps=TIMELINE_FPS,
        width=width,
        height=height,
        duration=(absolute_indices[-1] + 1) / TIMELINE_FPS,
        frames_indices=list(absolute_indices),
    )


def resize_frames(frames: np.ndarray, size: tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """`(T, H, W, 3) uint8` -> the 640x360 stack training fed the processor.

    Bilinear, and unconditionally to `size` without preserving aspect -- training
    resized the source frames the same way, so a clip that kept the source's aspect
    ratio squashes onto identical geometry.
    """
    if frames.shape[2] == size[0] and frames.shape[1] == size[1]:
        return frames
    return np.stack([
        np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR)) for f in frames
    ])


def clip_inputs(video_reader, start_time: float, end_time: float,
                size: tuple[int, int] = FRAME_SIZE, n_frames: int = N_FRAMES):
    """`(video_array, VideoMetadata)` for one question, from an open decord reader."""
    clip_fps = float(video_reader.get_avg_fps())
    if not clip_fps or clip_fps <= 0:
        raise ValueError(f"clip reports unusable fps {clip_fps!r}")

    clip_idx, absolute_idx = sample_indices(
        start_time, end_time, clip_fps, len(video_reader), n_frames)

    # One batched decode for every index at once, as the template does; decord
    # handles the repeated indices of a short clip without decoding twice.
    frames = video_reader.get_batch(clip_idx).asnumpy()  # (T, H, W, 3) uint8 RGB
    return resize_frames(frames, size), build_metadata(absolute_idx, size)
