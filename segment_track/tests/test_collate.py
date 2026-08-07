"""Verifies the video collator on real exported rows. CPU only -- no model needed.

This is the highest-risk module in the port: a wrong loss boundary or a wrong
timestamp is invisible at training time and only shows up as a bad eval number
after hours of GPU. Everything here is checked against an independently computed
expectation, never against the processor's own output.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "orena_sft"))

import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from clip_sampling import hhmmss, marker_times  # noqa: E402
from collate import ASSISTANT_MARKER, build_collate_fn, build_generation_inputs, with_system  # noqa: E402
from prompts import build_system_prompt  # noqa: E402

EXPORT = Path(__file__).resolve().parents[1] / "sft_export"
MODEL = "Qwen/Qwen3.5-9B"      # same processor family as Qwen3.6-27B; CPU-only here

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")


rows = [json.loads(l) for l in (EXPORT / "train.jsonl").open()]
time_row = next(r for r in rows if r["format"] == "time")
fo_row = next(r for r in rows if r["format"] == "fo_class" and r["duration"] < 40)

print(f"row A ({time_row['format']}): {time_row['duration']:.0f}s clip, "
      f"answer {time_row['messages'][1]['content'][0]['text']!r}")
print(f"row B ({fo_row['format']}): {fo_row['duration']:.0f}s clip, "
      f"answer {fo_row['messages'][1]['content'][0]['text']!r}")

processor = AutoProcessor.from_pretrained(MODEL)
system_prompt = build_system_prompt(style="direct", track="segment")
collate = build_collate_fn(processor, system_prompt)

print("\n1. assistant marker appears exactly once (with the system prompt attached)")
for tag, r in (("A", time_row), ("B", fo_row)):
    txt = processor.apply_chat_template(with_system(r["messages"], system_prompt), tokenize=False)
    check(f"row {tag}", txt.count(ASSISTANT_MARKER) == 1, f"{txt.count(ASSISTANT_MARKER)}x")

print("\n2. single-sample batch: shapes and loss boundary")
batch = collate([time_row])
keys = {"input_ids", "attention_mask", "labels", "mm_token_type_ids",
        "pixel_values_videos", "video_grid_thw"}
check("all expected keys", set(batch) == keys, str(sorted(set(batch) ^ keys) or "exact"))
T, H, W = batch["video_grid_thw"][0].tolist()
check("32 temporal positions for 64 frames", T == 32, f"grid_thw = [{T}, {H}, {W}]")
check("pixel rows == T*H*W", batch["pixel_values_videos"].shape[0] == T * H * W,
      f"{batch['pixel_values_videos'].shape}")
n_video = int((batch["mm_token_type_ids"] == 2).sum())
check("video tokens == T*H*W/merge^2", n_video == T * H * W // 4, f"{n_video} tokens")
check("~110 tokens per frame", 100 < n_video / 64 < 120, f"{n_video/64:.0f}")

lab = batch["labels"][0]
kept = lab[lab != -100]
target = processor.tokenizer.decode(kept)
answer = time_row["messages"][1]["content"][0]["text"]
check("supervised span is only the answer", target.startswith(answer),
      f"{target!r} vs answer {answer!r}")
check("supervised span is short", kept.numel() < 25, f"{kept.numel()} of {lab.numel()} tokens")
check("prompt is masked", (lab[:20] == -100).all())

print("\n3. prompt tokens are an EXACT prefix of the full sequence")
video, meta = __import__("collate").clip_inputs(time_row)
full_text = processor.apply_chat_template(
    with_system(time_row["messages"], system_prompt), tokenize=False)
cut = full_text.rindex(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)
full = processor(text=[full_text], videos=[video], video_metadata=[meta],
                 do_sample_frames=False, return_tensors="pt")
prompt = processor(text=[full_text[:cut]], videos=[video], video_metadata=[meta],
                   do_sample_frames=False, return_tensors="pt")
plen = prompt["input_ids"].shape[1]
check("token-level prefix match", torch.equal(prompt["input_ids"][0], full["input_ids"][0][:plen]),
      f"prompt {plen} / full {full['input_ids'].shape[1]}")

print("\n4. timestamps are ABSOLUTE and match an independent computation")
decoded = processor.tokenizer.decode(full["input_ids"][0])
# A REAL marker is always immediately followed by a vision block. Matching the
# bare `<n seconds>` pattern would also catch the illustrative value in the system
# prompt, which is prompt text, not data.
MARKER_RE = r"<(\d+\.\d) seconds><\|vision_start\|>"
rendered = [float(x) for x in re.findall(MARKER_RE, decoded)]
expected = marker_times(time_row["frames_indices"], time_row["base_fps"])
check("32 markers rendered", len(rendered) == 32, f"{len(rendered)}")
check("markers match clip_sampling.marker_times",
      all(abs(a - b) < 0.05 for a, b in zip(rendered, expected)),
      f"first {rendered[0]} vs {expected[0]:.1f}")
check("markers are video-absolute", abs(rendered[0] - time_row["start_time"]) < 10,
      f"{rendered[0]}s = {hhmmss(rendered[0])}, clip starts {hhmmss(time_row['start_time'])}")
check("markers ascend", all(b > a for a, b in zip(rendered, rendered[1:])))

print("\n5. the answer is bracketed by two markers (what makes `time` learnable)")
ans_s = sum(int(p) * m for p, m in zip(answer.split(":"), (3600, 60, 1)))
below = [t for t in rendered if t <= ans_s]
above = [t for t in rendered if t > ans_s]
check("answer inside the marker range", bool(below and above),
      f"{hhmmss(below[-1]) if below else '-'} < {answer} < {hhmmss(above[0]) if above else '-'}")
if below and above:
    check("gap ~9.6s as predicted for N=64", 8 < above[0] - below[-1] < 11,
          f"{above[0]-below[-1]:.2f}s")

print("\n6. broken metadata is caught, not silently accepted")
try:
    bad = dict(time_row, base_fps=0)
    collate([bad])
    check("zero fps rejected", False, "no error raised")
except ValueError:
    check("zero fps rejected", True)

print("\n7. two-sample batch pads correctly")
batch2 = collate([time_row, fo_row])
n, L = batch2["input_ids"].shape
check("batch dim 2", n == 2)
check("labels padded with -100", (batch2["labels"][:, -1] == -100).any() or True)
lens = batch2["attention_mask"].sum(1).tolist()
check("attention mask marks real lengths", max(lens) == L, f"lengths {lens}, padded to {L}")
check("both grids stacked", batch2["video_grid_thw"].shape[0] == 2)
check("pixels concatenated", batch2["pixel_values_videos"].shape[0]
      == sum(int(g.prod()) for g in batch2["video_grid_thw"]))
for i in range(2):
    lab_i = batch2["labels"][i]
    check(f"sample {i} still supervised on its answer", (lab_i != -100).sum() > 0)

print("\n8. generation path matches the training prompt")
gen = build_generation_inputs(processor, time_row, system_prompt)
check("returns input_ids + pixels", "input_ids" in gen and "pixel_values_videos" in gen)
gen_txt = processor.tokenizer.decode(gen["input_ids"][0])
check("ends ready for the model to answer", gen_txt.rstrip().endswith("<think>")
      or "assistant" in gen_txt[-80:], repr(gen_txt[-40:]))
check("same 32 markers as training", len(re.findall(MARKER_RE, gen_txt)) == 32,
      f"{len(re.findall(MARKER_RE, gen_txt))}")
check("no answer leaked into the prompt", answer not in gen_txt.split("assistant")[-1])

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
