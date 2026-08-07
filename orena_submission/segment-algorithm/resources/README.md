# `resources/` — what the image carries

Everything here is copied into the container image at build time and is all the
algorithm has at inference: **the container runs with networking disabled**, so
nothing can be downloaded at runtime.

| Path | What it is |
|:--|:--|
| `segment_model/` | the merged Qwen3.6-27B + LoRA-r8 SFT checkpoint, ~54.7 GB of bf16 safetensors in 7 shards |
| `prompts.py` | the system prompt + answer parser the checkpoint was trained with — a byte-for-byte copy of `orena_sft/prompts.py` |
| `clip_frames.py` | rebuilds training's 80-frame clip input, and its absolute-timestamp metadata, from the platform's trimmed 5 fps clip |

The template's `model.py` and `dummy_weights.pt` were removed: there is no custom
`nn.Module` to define, since the model loads through
`transformers.Qwen3_5ForConditionalGeneration.from_pretrained()` straight from
`segment_model/`.

## `segment_model/` is generated, not committed

It is written by `../reshard_weights.py` from the merged checkpoint at
`segment_track/checkpoints/segment-27b-alldata-n80-650-20260730-merged`:

```bash
venv3.12/bin/python orena_submission/segment-algorithm/reshard_weights.py \
    --src segment_track/checkpoints/segment-27b-alldata-n80-650-20260730-merged \
    --dst orena_submission/segment-algorithm/resources/segment_model
```

**Why re-shard at all.** A single image layer cannot exceed 50 GB and each `COPY`
is exactly one layer. As merged, the checkpoint's larger shard was 49.83 GB —
inside the limit by 0.35%, which is not a margin worth risking a 55 GB upload on.
Re-sharded to ~8 GB, the `Dockerfile`'s two globbed `COPY`s produce layers of about
32 GB and 23 GB.

## Why the prompt is reproduced rather than rebuilt

`prompts.py` is copied verbatim so the container prompts the model exactly as
training did — `build_system_prompt(include_definitions=False, style="direct",
track="segment")`. The FO class names come from `FOType.names()` in
`orena-focus==0.3.5`, which is why that version is pinned in `requirements.txt`.

The platform also supplies `FO_definitions.json` per run. It is deliberately *not*
folded into the prompt: this checkpoint was trained against the static registry
baked into the package, and swapping in a different class list would prompt it in a
shape it never saw. If the test phase introduces new FO classes, the model will not
know them — a modelling gap, not a packaging bug.

## Why `clip_frames.py` exists

Training never decoded a video. It read pre-extracted JPEGs of the **source**
procedure video and gave the processor absolute frame indices, which is what makes
Qwen3-VL render `<1234.5 seconds>` markers on the source timeline — the anchor the
system prompt tells the model to read `time` answers off (38.5% of segment
questions). The platform instead hands over a clip already trimmed to the window at
5 fps, whose own timeline starts at zero. `clip_frames.py` restores the absolute
timeline and picks the nearest available frame for each instant training would have
sampled. `../test/verify_timeline.py` checks that against all 6254 exported test
records.
