"""Verifies the procedure collator through the REAL processor. CPU only, no model.

Builds its records directly from a video on disk rather than from `sft_export/`, so it
runs before the export exists and does not go stale when the export is rebuilt.

This is the highest-risk module in the track: the whole point is that
`frames_indices` are 5 fps indices while the JPEGs are at native fps, and getting that
backwards renders every timestamp 5x off -- invisible until the eval numbers come back
wrong. Everything is checked against an independently computed expectation, never
against the processor's own output.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "segment_track"))

import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from collate_procedure import (  # noqa: E402
    ASSISTANT_MARKER, build_collate_fn, build_generation_inputs, clip_inputs, with_system,
)
from prompts_procedure import build_system_prompt, window_line  # noqa: E402
from sampling import FPS, sample_indices_5fps, to_native_index  # noqa: E402

MODEL = "Qwen/Qwen3.5-9B"
FRAME_DIR = "/projects/datasets_ML/orena/heico/frames_overlay/0009 - Heico - Prokto - 10"
BASE_FPS = 25.0
MARKER_RE = r"<(\d+\.\d) seconds><\|vision_start\|>"

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


def make_record(end_time, question, answer, fmt, n_max=64):
    idx5 = sample_indices_5fps(0.0, end_time, question, n_max=n_max)
    return {
        "uid": "heico/test", "format": fmt, "base_fps": BASE_FPS, "frame_dir": FRAME_DIR,
        "start_time": 0.0, "end_time": end_time, "duration": end_time,
        "frames_indices": idx5, "n_frames": len(idx5),
        "n_quoted_timestamps": 0,
        "messages": [
            {"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": f"{window_line(end_time)} {question}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
    }


# A 40-minute prefix (stretched) and a 4-minute one (true 5 s density), so both
# branches of the sampler go through the processor.
time_row = make_record(2400.0, "At what time was a Clip first visible in the video?",
                       "00:21:40", "time")
short_row = make_record(240.0, "Which foreign object classes appear in this video?",
                        "Clip, Sponge", "fo_class")

print(f"row A ({time_row['format']}): {time_row['duration']/60:.0f} min prefix, "
      f"{time_row['n_frames']} frames")
print(f"row B ({short_row['format']}): {short_row['duration']/60:.0f} min prefix, "
      f"{short_row['n_frames']} frames")

processor = AutoProcessor.from_pretrained(MODEL)
system_prompt = build_system_prompt(style="direct")
collate = build_collate_fn(processor, system_prompt)

print("\n1. the frames it loads are the NATIVE-fps ones the 5 fps indices point at")
video, meta = clip_inputs(time_row)
check("one frame per index", video.shape[0] == time_row["n_frames"], str(video.shape))
check("resized to 640x360", video.shape[1:] == (360, 640, 3), str(video.shape[1:]))
check("metadata fps is 5.0, not the source fps", meta.fps == FPS, str(meta.fps))
check("metadata carries the 5 fps indices", list(meta.frames_indices) == time_row["frames_indices"])
native0 = to_native_index(time_row["frames_indices"][1], BASE_FPS)
check("native lookup scales by base_fps/5",
      native0 == round(time_row["frames_indices"][1] * BASE_FPS / FPS),
      f"idx5 {time_row['frames_indices'][1]} -> native {native0}")

print("\n2. assistant marker appears exactly once")
for tag, r in (("A", time_row), ("B", short_row)):
    txt = processor.apply_chat_template(with_system(r["messages"], system_prompt), tokenize=False)
    check(f"row {tag}", txt.count(ASSISTANT_MARKER) == 1, f"{txt.count(ASSISTANT_MARKER)}x")

print("\n3. single-sample batch: shapes and loss boundary")
batch = collate([time_row])
keys = {"input_ids", "attention_mask", "labels", "mm_token_type_ids",
        "pixel_values_videos", "video_grid_thw"}
check("all expected keys", set(batch) == keys, str(sorted(set(batch) ^ keys) or "exact"))
T, H, W = batch["video_grid_thw"][0].tolist()
check("temporal positions are n_frames/2", T == time_row["n_frames"] // 2,
      f"grid_thw = [{T}, {H}, {W}]")
check("pixel rows == T*H*W", batch["pixel_values_videos"].shape[0] == T * H * W)
n_video = int((batch["mm_token_type_ids"] == 2).sum())
check("video tokens == T*H*W/merge^2", n_video == T * H * W // 4, f"{n_video} tokens")
check("~110 tokens per frame", 100 < n_video / time_row["n_frames"] < 120,
      f"{n_video/time_row['n_frames']:.0f}")

lab = batch["labels"][0]
kept = lab[lab != -100]
target = processor.tokenizer.decode(kept)
answer = time_row["messages"][1]["content"][0]["text"]
check("supervised span is only the answer", target.startswith(answer),
      f"{target!r} vs {answer!r}")
check("supervised span is short", kept.numel() < 25, f"{kept.numel()} of {lab.numel()}")
check("prompt is masked", (lab[:20] == -100).all())

print("\n4. prompt tokens are an EXACT prefix of the full sequence")
full_text = processor.apply_chat_template(
    with_system(time_row["messages"], system_prompt), tokenize=False)
cut = full_text.rindex(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)
full = processor(text=[full_text], videos=[video], video_metadata=[meta],
                 do_sample_frames=False, return_tensors="pt")
prompt = processor(text=[full_text[:cut]], videos=[video], video_metadata=[meta],
                   do_sample_frames=False, return_tensors="pt")
plen = prompt["input_ids"].shape[1]
check("token-level prefix match",
      torch.equal(prompt["input_ids"][0], full["input_ids"][0][:plen]),
      f"prompt {plen} / full {full['input_ids'].shape[1]}")

print("\n5. markers are ABSOLUTE procedure time, computed independently")
decoded = processor.tokenizer.decode(full["input_ids"][0])
rendered = [float(x) for x in re.findall(MARKER_RE, decoded)]
idx5 = time_row["frames_indices"]
expected = [(idx5[i] / FPS + idx5[i + 1] / FPS) / 2 for i in range(0, len(idx5) - 1, 2)]
check("one marker per fused pair", len(rendered) == len(idx5) // 2, str(len(rendered)))
check("markers match idx5 / 5.0",
      all(abs(a - b) < 0.05 for a, b in zip(rendered, expected)),
      f"first {rendered[0]} vs {expected[0]:.1f}")
check("markers start near the procedure start", rendered[0] < 60,
      f"{rendered[0]}s — the window is a prefix, so it begins at 0")
check("markers reach the window end",
      abs(rendered[-1] - time_row["end_time"]) < 2 * (time_row["end_time"] / len(rendered)),
      f"{rendered[-1]}s vs end {time_row['end_time']}s")
check("markers ascend", all(b > a for a, b in zip(rendered, rendered[1:])))
check("NOT 5x off (the fps trap)", rendered[-1] > time_row["end_time"] / 2,
      f"{rendered[-1]}s — passing base_fps instead of 5.0 would give "
      f"~{rendered[-1]/5:.0f}s")

print("\n6. broken metadata is caught, not silently accepted")
try:
    collate([dict(time_row, base_fps=0)])
    check("zero base_fps rejected", False, "no error raised")
except (ValueError, ZeroDivisionError, OSError):
    check("zero base_fps rejected", True)

print("\n7. two-sample batch with DIFFERENT frame counts pads correctly")
check("the two rows really do differ in length",
      time_row["n_frames"] != short_row["n_frames"],
      f"{time_row['n_frames']} vs {short_row['n_frames']}")
batch2 = collate([time_row, short_row])
n, L = batch2["input_ids"].shape
check("batch dim 2", n == 2)
lens = batch2["attention_mask"].sum(1).tolist()
check("attention mask marks real lengths", max(lens) == L, f"lengths {lens}, padded to {L}")
check("shorter row is genuinely padded", min(lens) < L, f"{min(lens)} < {L}")
check("both grids stacked", batch2["video_grid_thw"].shape[0] == 2)
check("pixels concatenated", batch2["pixel_values_videos"].shape[0]
      == sum(int(g.prod()) for g in batch2["video_grid_thw"]))
for i in range(2):
    check(f"sample {i} still supervised on its answer", (batch2["labels"][i] != -100).sum() > 0)

print("\n8. generation path matches the training prompt")
gen = build_generation_inputs(processor, time_row, system_prompt)
gen_txt = processor.tokenizer.decode(gen["input_ids"][0])
check("returns input_ids + pixels", "input_ids" in gen and "pixel_values_videos" in gen)
check("same markers as training", len(re.findall(MARKER_RE, gen_txt)) == len(rendered))
check("no answer leaked into the prompt", answer not in gen_txt.split("assistant")[-1])
check("window line present", "Procedure observed from 00:00:00 to 00:40:00" in gen_txt)

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
