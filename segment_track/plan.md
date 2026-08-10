# SEGMENT-track SFT pipeline for Qwen3.6-27B — plan and as-built record

Port of the frame-track method ([method_frame.md](../orena_sft/method_frame.md)) to
the **SEGMENT** track (video clips up to 5 min) with **`Qwen/Qwen3.6-27B`**. Same
shape as the frame pipeline — build a JSONL export, LoRA-SFT with TRL under DDP,
evaluate with the FOCUS `Evaluator` — but four things genuinely change: the visual
input is a *clip*, the question mix is dominated by **absolute-timestamp** answers,
the model is **3× larger**, and only **2 H200s** exist on this cluster.

**The pipeline is built and has shipped a model.** §§0–2 are the design rationale,
updated to the numbers that were actually measured rather than the ones estimated
before the code existed. §3 records what was built; §5 records the final run,
`segment-27b-alldata-n80-650-20260730`, which is the model to submit. Where a
prediction from the original plan was wrong, it is corrected in place and the
correction is called out — those are the parts worth reading twice.

All segment-track code lives in this folder. Shared, track-agnostic pieces
(`prompts.py`, `frame_path()`) stay in `../orena_sft/` as the single source of truth
and are imported via `sys.path.insert`, the pattern the repo already uses — see §2.5.

---

## 0. What is actually different from the frame track

| | FRAME (done earlier) | SEGMENT (this pipeline) |
|---|---|---|
| visual input | 1 JPEG | N sampled frames from `[start_time, end_time]` |
| clip length | 0 s | median **119 s**, p25 29 s, max **300 s** |
| dominant format | `fo_class` 46% | **`time` 38.5%** (absolute `hh:mm:ss`) |
| new formats | — | `percentage` (42 rows), much more `multiple_choice` |
| capabilities | 2 groups (recognition, aggregation) | all 5 groups incl. temporal grounding, event/procedural, complex reasoning |
| latency budget | 5 s/question | **15 s/question** (`TRACK_MAX_LATENCY[Track.SEGMENT]`) |
| model | Qwen3.5-9B, 2×H200, bs 16/GPU | Qwen3.6-27B, 2×H200, bs 1/GPU + grad-accum |

**The single hardest thing** is `time`: the answer is an *absolute* video timestamp
(e.g. `01:09:58`) while the model only sees a clip window. Tolerance is
`min(5.0, 1 + duration·4/360)` — for the median 119 s clip that is **2.3 s**. See
§2.3.

**Shipped configuration:** Qwen3.6-27B · **N=80 frames** @ 640×360 · 2×H200 ·
LoRA r=8 α=16 on 7 text projections (39.8 M params, 0.145%) · batch 1 × grad-accum
16 (effective 32) · gradient checkpointing · lr 1e-4 · **650 steps ≈ 1.04 epochs**.
Measured: 74 GB peak VRAM, 81.7 s/step, 14.8 h wall. Rationale in §2.1, §2.3 and
§2.8; the run itself in §5.

---

## 1. Facts established (verified, not assumed)

### 1.1 Data — segment track, from the cached HF parquet

| dataset / split | rows | videos | procedure |
|---|---|---|---|
| heico train | 8,000 | 20 | Proctocolectomy, Rectal Resection |
| heico test | 4,000 | 10 | **Sigmoid Resection** (unseen procedure) |
| lapchole train | 5,746 | 72 | Lap. Cholecystectomy |
| lapchole test | 2,254 | 28 | Lap. Cholecystectomy |
| **combined train** | **13,746** | **92** | |
| **combined test** | **6,254** | **38** | |

Answer formats, combined train: `time` 5,299 · `fo_class` 3,457 ·
`multiple_choice` 2,240 · `binary` 945 · `number` 893 · `open_ended` 870 ·
`percentage` 42.

Row schema is identical to the frame track (`id, video, procedure_type, question,
answer, answer_format, track, generation, clinical_relevance, ood,
timestamp_start, timestamp_end, primary_capability, secondary_capabilities`), so
`FocusDataset(dataset, split, Track.SEGMENT)` parses it with no changes.
`primary_capability` comes through as leaf codes (`2a`, `1a`, `1d`, …) resolved by
`Capability.from_any`; `ood` is `False` everywhere in train.

Clips overlap heavily: consecutive rows are the *same* question anchored to
sliding windows over the same video. This makes a **video-level split
mandatory** (an even stronger requirement than in the frame track) and means
near-duplicate augmentation is already baked into the data.

### 1.2 Frames are already on disk — no video decoding needed

Every frame of every relevant video is already extracted as JPEG at native fps:

- `/projects/datasets_ML/orena/{heico,lapchole}/frames/<video_stem>/frame{idx:07d}.jpg`
- heico: 30 videos @ 25 fps (e.g. 371,603 frames for one video); lapchole: 100
  videos @ 30 fps — exactly the 72 train + 28 test segment videos
- **960×540 RGB**, ~80 KB/frame — this is also the videos' native resolution

So a clip is just a list of frame indices, resolved with the frame track's
existing `frame_path()` helper (`round(t · base_fps)`). This avoids
`FocusVideoDataset` entirely at training time — that class re-encodes a temp MP4
per sample via `decord` + `cv2.VideoWriter`, which would be the throughput
bottleneck of the whole run. **Decision: sample JPEGs, never decode video.**

Timestamp-overlaid copies exist at the same coverage (`overlayed/`,
`frames_overlay/`: heico 30/30, lapchole 100/100) — see §2.3(b).

### 1.3 The model

`Qwen/Qwen3.6-27B` — ~54 GB bf16, cached locally (`preflight.py` §2 asserts it).
The config declares:

```json
"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
"video_token_id": 248057, "vision_config": {"depth": 27, "out_hidden_size": 5120, ...}
"text_config": {"hidden_size": 5120, "num_hidden_layers": 64, "max_position_embeddings": 262144}
```

Consequences:
- **The installed `transformers` 5.9.0 can load it** with the same
  `Qwen3_5ForConditionalGeneration` class and `AutoProcessor` the frame script
  uses. No library upgrade, no `trust_remote_code`. (There is no `qwen3_6` module
  in transformers — and none is needed.)
- Weights must be downloaded first: **~54 GB** bf16 (`Qwen3.5-27B` is already
  cached at 52 GB, and `/home/ajenane` has 347 GB free).
- Hybrid attention (`linear_attention` ×3 / `full_attention` every 4th layer,
  `full_attention_interval: 4`) — long video contexts are cheaper here than in a
  dense model, which is the reason this track is worth a 27B at all.
- The processor is `Qwen3VLProcessor` with a `Qwen3VLVideoProcessor`
  (`video_preprocessor_config.json` present), `temporal_patch_size: 2`,
  `merge_size: 2`, video pixel budget `longest_edge: 25165824`.

### 1.4 Hardware — the real constraint

`sinfo` shows **one** GPU node for this work: `gpu38`, partitions
`gpu-large` / `gpu-large-interactive`, `gpu:h200:2` + `gpu:rtx_pro_6000:4`,
770 GB RAM. 27B bf16 weights (54 GB) + LoRA optimizer state + activations fit on a
141 GB H200, so **DDP with 2 replicas still works** — but per-device batch drops to
1–2 and gradient checkpointing becomes necessary. The 4× RTX PRO 6000 (96 GB) are a
viable second config if the H200s are busy.

---

## 2. Design decisions — and how each one held up

### 2.1 Clip → frames: the sampling policy

Fixed **N frames uniformly spanning `[start_time, end_time]`**, not fixed fps.
Rationale: durations span 1 s → 300 s (300×), so fixed fps makes sequence length
vary by 300× and batching impossible; fixed N makes every sample the same token
cost, which is what lets `--batch-size` and the memory ceiling be predictable.

**Decision: flat N=80 @ 640×360** (`clip_sampling.DEFAULT_N_FRAMES`). Two
constraints fix the shape, both measured (§2.7):

- **N must be even.** `temporal_patch_size=2` fuses frames in pairs; an odd N
  leaves a ragged pair.
- **Temporal resolution is N/2, not N.** The pairing means 80 frames give the model
  **40** time-anchors, not 80 — see §2.3 for what that costs.

Measured token cost (verified against the real processor, not estimated):

| frames | temporal positions | video tokens | tok/frame |
|---|---|---|---|
| 32 @ 640×360 | 16 | 3,520 | 110 |
| 64 @ 640×360 | 32 | 7,040 | 110 |
| **80 @ 640×360** | **40** | **8,800** | **110** |

With the `direct` system prompt and a question, the full prompt is **9,915 tokens**
(logged by `preview_example` on a real row).

The plan originally committed to N=64 on the anchor-spacing argument in §2.3. N=80
was taken instead once §2.8's memory prediction proved pessimistic — 74 GB of 141
measured — so the extra anchors were free. **This is the one place the reasoning did
not pay off:** §2.3's model says N=80 puts the anchor grid inside tolerance for
*every* clip length, yet the shipped model scores 0.49 / 0.62 on `time` (§5). Anchor
density is therefore **not** the binding constraint, and the N=140 export
(`sft_export_n140/`) exists to test that — see §4.

*Optional refinement, not adopted:* an adaptive `N = min(2·ceil(duration/4), 80)`
would cut short clips' cost with no accuracy loss (a 29 s clip needs nowhere near
40 anchors). It saves **time, not memory** — peak VRAM is still set by the 300 s
clips that hit the cap — and it makes step time variable. Never became binding.

### 2.2 Video input, not a batch of images

Pass the sampled frames to the processor as `videos=[frames]` (one video), not as
N separate images. Two reasons: the chat template renders
`<|vision_start|><|video_pad|><|vision_end|>` and the processor expands it into
**per-frame timestamp markers** (§2.3); and the video path applies
`temporal_patch_size=2`, halving the token count versus N independent images.

This changes the collator's tensor contract versus the frame script:
`pixel_values` / `image_grid_thw` become **`pixel_values_videos` /
`video_grid_thw`**, and `mm_token_type_ids` must mark video tokens. Everything
else about the frame collator — render the full chat, find the literal
`<|im_start|>assistant\n<think>\n\n</think>\n\n` marker with `rindex`, slice,
`labels[:prompt_len] = -100` — carries over verbatim and should be reused, not
reinvented.

### 2.3 Absolute time: two anchor mechanisms

**First, the thing that dominates both of them: anchor density.** No timestamp
representation lets the model read an answer off a frame — it must interpolate
between anchors either way, so what matters is how far apart the anchors are.

**The anchors are N/2, not N** (§2.7): frames are fused in pairs, and one
timestamp marker is emitted per pair. Verified on a real 299 s clip with 32
frames — the markers came out **19.3 s** apart (`559.8, 579.1, 598.4, …`), and the
ground-truth answer `00:10:12` (612.0 s) fell between `598.4` and `617.7`. A 19.3 s
window for a ±4.3 s target.

Landing inside the tolerance means anchor spacing ≤ 2× tolerance:

| N | anchors | spacing on a 299 s clip | tolerance | predicted hit rate |
|---|---|---|---|---|
| 32 | 16 | 19.3 s | ±4.3 s | ~45% |
| 64 | 32 | 9.6 s | ±4.3 s | ~90% |
| **80** | **40** | **7.7 s** | **±4.3 s** | **covered** |

On the median 119 s clip N=80 gives 3.1 s spacing against a ±2.3 s tolerance, and on
the worst case (299 s) 7.7 s against ±4.3 s — an 8.6 s window. By this model the
anchor grid is sufficient at every clip length shipped.

**It was not sufficient in practice.** The shipped model scores 0.49 (heico) / 0.62
(lapchole) on `time`, and `duration_estimation` — which reads *two* anchors and
subtracts — sits at 0.29. So the residual error is not anchor spacing; it is the
model interpolating between anchors and converting to `hh:mm:ss`. Treat the table
above as a **necessary condition that is now satisfied**, not as a predictor of
accuracy. The anchor mechanism below is second-order to it either way.

**(a) Processor timestamp markers.** `Qwen3VLProcessor.__call__` builds the video
placeholder as, per frame, `<{t:.1f} seconds>` + vision block, where `t` comes from
`video_metadata.frames_indices` and `video_metadata.fps`
(`transformers/models/qwen3_vl/processing_qwen3_vl.py:151`, `video_utils.py:100`).
Passing `VideoMetadata(fps=base_fps, frames_indices=<absolute frame indices>,
total_num_frames=<len of source video>)` gives the model absolute video timestamps
interleaved with the frames, natively — no pixel cost, exact, and the format
Qwen3-VL was pretrained on.

**(b) Burned-in overlay.** `frames_overlay/` (heico 30/30 videos, lapchole 100/100,
same frame counts as `frames/`) has `hh:mm:ss` drawn by
`VideoTimestampOverlayPreprocessor`: `cv2.putText`, white, HERSHEY_SIMPLEX
scale 1.5, thickness 3, at (20,50), on the native 960×540. Verified: the clock sits
in the **black letterbox** outside the endoscopic circle — white on black, never
occluding tissue — and stays legible at 640×360 and even 480×270. One flag to use
it: `FocusConfig(frames_folder="frames_overlay")`.

| | (a) markers | (b) overlay |
|---|---|---|
| anchor accuracy | exact, text tokens | OCR — small but nonzero error |
| units | `<4198.0 seconds>` → needs divmod to `hh:mm:ss` | already `hh:mm:ss`, no arithmetic |
| pixel / token cost | none | none (dead letterbox pixels) |
| in-distribution for Qwen3-VL | yes | incidental |
| at submission time | free — we control `video_metadata` | **we must burn the clock ourselves** on raw challenge video with byte-identical draw params |

**Verified: the two clocks agree exactly.** The overlay computes
`int(frame_idx / fps)` from the video's own fps while `frame_path()` uses the
hardcoded `DATASET_BASE_FPS`; the real files are exactly 25.0 (heico) and 30.0
(lapchole), matching the constants. A 29.97-vs-30 mismatch would have drifted
~3.6 s/hour — past tolerance, and silently. **Assert this in the builder** rather
than trusting it to stay true.

**Decision: markers are the default.** Three of the four rows above favour them,
and the submission-time row is the least reversible — the overlay would make
burning a pixel-exact clock a dependency of the submission container, with a silent
failure mode if any draw parameter drifts.

The one thing markers do *not* give is the format match: they hand the model an
exact value in the wrong units, and the divmod to `hh:mm:ss` is exactly where a
2.3 s budget gets blown by an off-by-one-minute slip.

**Settled: markers only; the overlay arm was never built.** The standalone
`probe_timestamp_markers.py` gate was folded into the base evaluation instead — it
answers the same question on the whole test split rather than 50 rows. Base
Qwen3.6-27B scores 0.21 / 0.18 on `time` with markers alone, which is poor but is
*non-compliance*, not blindness: after SFT the same plumbing reaches 0.49 / 0.62
with no change to the anchor mechanism. The divmod is learnable, so the overlay's
only advantage never became worth its submission-time cost. `--frames-folder
frames_overlay` remains a working flag on the builder if that changes.

Two failure modes to guard:
1. **Never let `fps` be inferred.** If metadata is missing the processor warns and
   defaults to `fps=24`, silently producing wrong timestamps. Assert
   `video_metadata` is populated in the collator, and unit-check that the rendered
   text contains the expected `<… seconds>` values.
2. **Overlay draw params must be identical at inference.** Font, scale, thickness
   and position are a train/inference contract; a mismatch is silent. If the
   overlay arm is built, pin those constants in one shared place used by both the
   builder and the submission wrapper.

Remaining ablation arms (all export-time flags, cheap): absolute vs. clip-relative
`frames_indices`; `frames/` vs. `frames_overlay/`; and — orthogonal to both —
prepending `Clip window: 01:08:00–01:12:59.` to the question, which gives the
anchor in the answer's own format with no pixels and no metadata at all.

### 2.4 Prompt: extend `../orena_sft/prompts.py`, don't fork it

The frame prompt already covers binary / number / fo_class / time /
multiple-choice / open-ended shapes. Segment needs:
- a **`percentage`** rule (42 rows, `threshold_pp=5.0`) — currently absent
- an explicit **`hh:mm:ss` from the `<… seconds>` markers** instruction (§2.3)
- comma-separated multi-timestamp answers (`Time.verify` accepts them)
- wording that acknowledges a clip rather than "one frame … from a laparoscopic
  procedure"

Add a `track` parameter to `build_system_prompt()` rather than copying the module,
so the FO vocabulary stays read live from `FOType.names()` and the frame arm is
untouched. `extract_answer()` is track-agnostic and is reused as-is.

Keep the same arms as the frame runs (`plain` / `direct` / `--system-prompt-file`)
so the GEPA loop in [../orena_gepa/](../orena_gepa/) can later optimize a segment
prompt with no new machinery. `structured` stays refused — segment targets are
bare answers too.

### 2.5 Folder layout and shared code

As built:

```
segment_track/
  plan.md                          ← this file
  clip_sampling.py                 clip → indices, pixels, VideoMetadata   (§2.1, §2.3)
  collate.py                       records → model inputs, loss masking    (§2.2, §2.7)
  build_segment_sft_dataset.py     export → train/eval/test.jsonl
  build_stratified_subsample.py    test_strat2000.jsonl for checkpoint sweeps
  sft_train_qwen_segment_ddp.py    LoRA SFT, DDP
    sft_train_qwen_segment_ddp.slurm   2×H200, the 27B arm
    sft_train_segment_9b_rtx.slurm     2×RTX PRO 6000, the 9B control arm
  evaluate_base_segment.py         the evaluator actually used: shardable,
    evaluate_base_segment.slurm    resumable, base OR --checkpoint-dir
  evaluate_qwen_segment.py         single-GPU variant; superseded, see §4
  merge_lora.py / .slurm           adapter → standalone weights, verified merge
  preflight.py                     CPU gate for the silent failures (§3)
  tests/                           clip_sampling · collate · export · train config
  sft_export/{train,eval,test,all,test_strat2000}.jsonl
  sft_export_n140/                 N=140 export, base-evaluated only (§4)
  checkpoints/<run_name>/
  logs/
```

`clip_sampling.py` and `collate.py` were not in the original file list and are the
two that matter most: every consumer — builder, trainer, generation callback, both
evaluators — goes through them, so training and inference cannot disagree about
what the model was shown or what time it is in the video.

Not built: `probe_timestamp_markers.py` (folded into the base eval, §2.3) and
`method_segment.md` (§4).

Two things stay in `../orena_sft/` and are imported, not copied:
- **`prompts.py`** — one FO vocabulary and one answer-rule block for both tracks
  (§2.4). Copying it would let the two drift, which is exactly the failure the
  frame pipeline's `with_system()` was written to prevent.
- **`frame_path()`** from `build_frame_sft_dataset.py` — the
  `round(t · base_fps)` formula must match `FocusFrameDataset` and must not be
  re-derived in a second place.

Both are reached with `sys.path.insert(0, <repo>/orena_sft)`, the pattern
`evaluate_qwen_frame.py` already uses. If a third track ever appears, promote them
to a shared package instead.

### 2.6 Frame I/O: read from `/projects`, hide the latency

Measured on the `/projects` NFS mount (960×540 JPEG, decode + resize to 640×360):

| access pattern | per frame | 32-frame clip |
|---|---|---|
| cold, scattered | 71 ms | **2.3 s** |
| warm (page cache) | 9.2 ms | 0.3 s |

The cold path is **latency-bound, not bandwidth-bound** — 2 MB/s effective means
one NFS round-trip per small file, nowhere near saturating the link.

**Decision: leave the frames on `/projects` and read them at load time.** Staging
is the wrong instinct here, and not because of the 3.6 TB total (heico 1.6 +
lapchole 2.0) — the shipped export touches ≈128 GB (20k rows × 80 × ~80 KB, before
the heavy frame reuse between overlapping windows). It is wrong because copying
~1.6 M small files *is itself* ~1.6 M round-trips. The data is fine where it is;
what has to change is how many reads are in flight.

1. **Parallel dataloader workers — the actual fix.** The latency is per-file and
   embarrassingly parallel. One optimizer step is 16 micro-batches × 80 frames =
   1,280 frames per GPU: 91 s single-threaded, ~2.8 s across the 32 CPUs the SLURM
   script already requests. A 27B fwd+bwd at ~9.9k tokens × 16 accumulation steps
   measured 81.7 s, so loading hides completely behind compute. Set
   `dataloader_num_workers` to 16–24 with `persistent_workers=True`.
2. **The page cache stages itself, for free.** Overlapping windows mean the *distinct*
   frame set is far smaller than the 128 GB of reads, so against a 256 GB `--mem`
   request the hot frames become resident within the first epoch and reads drop to
   9 ms — no code, no copying.
3. **Only if the smoke run shows starvation:** repack at export time into one file
   per clip on node-local scratch, pre-resized to the training resolution. That turns
   80 round-trips per sample into one sequential read and drops the decode cost too.
   Real engineering — justify it with a measured profile, not an assumption.

**Outcome: 1 and 2 sufficed; the repack in 3 was never needed.** Training runs with
`--dataloader-workers 16` and `persistent_workers=True`, and at 81.7 s/step the
80-JPEG read never surfaced as a stall. Inference needed one addition the plan did
not anticipate: generation is a single sequential process with no dataloader, so
`evaluate_base_segment.warm_page_cache()` prefetches the *next* clip's frames on an
8-thread pool while the current one generates. Without it the cold read (8.5 s for
80 frames, vs 0.5 s warm) lands inside `prep_time`, which the 15 s ceiling is judged
on. Note this makes the reported latency assume locally-resident frames — see §5.

### 2.7 How the processor actually works — traced, not assumed

Run on CPU with the cached `Qwen3.5-9B` processor (`Qwen3VLProcessor` +
`Qwen3VLVideoProcessor` — structurally identical to what Qwen3.6-27B declares) on a
real row: heico `0009 - Heico - Prokto - 10.avi`, clip `00:09:15`–`00:14:14`
(299 s), question *"There is one Sponge in the frame at 00:09:19. When is it
retrieved…"*, answer `00:10:12`.

**Input:** 32 frames sampled at indices `13875 … 21350` (one per 9.65 s), loaded
from JPEG, resized to 640×360, stacked → `(32, 360, 640, 3) uint8`.

**What the processor returned:**

```
video_grid_thw       = [16, 22, 40]
pixel_values_videos  = (14080, 1536)
input_ids            = (1, 3771)   →  3520 video + 251 text tokens
```

Reading the grid:
- **space** — 640×360 in 16×16 patches = 40×22 = 880/frame; a 2×2 spatial merge
  gives **220 tokens per frame**
- **time** — `temporal_patch_size=2` fuses frame *pairs* → **16** positions, not 32
- reconciles: 16×22×40 = 14,080 patch rows, each 1536 = 3 ch × 16 × 16 px × 2
  frames; tokens 16 × 220 = 3,520

**Timestamps are injected into the text**, one marker per fused pair:

```
<|im_start|>user
<559.8 seconds><|vision_start|>[…220 video tokens…]<|vision_end|>
<579.1 seconds><|vision_start|>[…]<|vision_end|>
… 16 markers total …
<849.2 seconds><|vision_start|>[…]<|vision_end|>
There is one Sponge in the frame at 00:09:19. …<|im_end|>
```

`559.8 s = 00:09:19`, `849.2 s = 00:14:09` — **absolute source-video time**, which
is exactly the answer format's frame of reference. (Markers are the *mean* of their
fused pair, hence 559.8 rather than the clip start 555.0.)

**Loss masking verified on the same sample:**

```
prompt tokens 3761 | trained-on 10 | prefix check OK
target decodes to '00:10:12<|im_end|>\n'
```

10 supervised tokens out of 3,771. The marker-slice approach from the frame
trainer transfers unchanged.

#### Two mandatory API details, both found by this trace

1. **`do_sample_frames=False` is required.** By default the video processor
   *re-samples* the array it is given, using indices computed from
   `metadata.total_num_frames` — on pre-sampled frames that raises
   `IndexError: index 484 is out of bounds for axis 0 with size 32`. A crash is the
   lucky outcome; with `total_num_frames` set to the frame count instead, it would
   silently resample and **overwrite `frames_indices`**, destroying the absolute
   timestamps.
2. **`VideoMetadata` must carry real `fps` and absolute `frames_indices`.** Tested
   the failure mode directly — omit them and the processor warns once, defaults to
   `fps=24`, and renders `<0.0 seconds> … <1.3 seconds>`: the model is told a
   5-minute clip spans 1.3 s while being asked for an answer at `00:10:12`.
   Training would proceed and produce garbage. **Assert, don't trust.**

### 2.8 Budgets — predicted, then measured

**Training, per GPU, batch 1, LoRA r=8 + gradient checkpointing:**

| | predicted (N=64, ~7.3k tok) | **measured (N=80, 9.9k tok)** |
|---|---|---|
| peak VRAM | ~80–85 GB of 141 | **74.0 GB of 141** |
| per optimizer step | ~60–80 s | **81.7 s** |

The memory prediction was ~10 GB pessimistic *at a larger N* — the activation and
logits estimates were both high. That headroom is what allowed N=64 → N=80 (§2.1).
Chunked cross-entropy, the escalation lever for the 248k-vocab logits tensor, was
never needed.

Batch 1 also measured **faster** than batch 2 (66 s vs 71 s/step in the timing runs)
once the fused `fla` kernels removed the launch-overhead bottleneck, so larger
per-device batches now only add memory pressure. The step time is fla-dependent, not
incidental: without `flash-linear-attention` the 48 linear-attention layers fall back
to a Python loop at **261 s/step, 4× slower**. `preflight.py` and
`check_fast_kernels()` both gate on this, and on Triton ≥ 3.7.1, which below that
version computes *wrong gradients* on Hopper (fla #640).

**Wall clock**, 625 optimizer steps/epoch (20,000 ÷ 32) on the final export:

| | per step | 650 steps (1.04 epochs) |
|---|---|---|
| 2×H200, N=80 | 81.7 s | **14.8 h** |

Comfortably inside the 2-day SLURM limit the `.slurm` wrapper requests. The original
"1.5 epochs / 23 h" framing is obsolete: step count, not epochs, became the control
(§5).

**Inference, measured on the shipped model** over all 6,254 test clips, one RTX PRO
6000, N=80 — against the 15 s SEGMENT ceiling:

| | median | p95 | p99 | max | over 15 s |
|---|---|---|---|---|---|
| SFT 27B | 2.89 s | 3.16 s | 3.22 s | 4.32 s | **0** |

~4× headroom, and the distribution is tight — the plan's ~5 s estimate was close.
The KV cache is negligible because only **16 of 64 layers** use full attention, with
4 KV heads. Base-model runs show a fatter tail (p999 8.4 s, max 14.0 s) that is I/O
noise, not compute; see §2.6 on the prefetch.

*Verified in §5:* `Evaluator.run(track=Track.SEGMENT)` scores violations as **wrong**
rather than erroring, so a latency regression would look like an accuracy drop. Zero
violations occurred in any arm.

**Inference, against the challenge runtime (80 GB GPU, 15 s/question):**

| VRAM | | latency (A100 80GB) | |
|---|---|---|---|
| weights (LoRA merged) | 54 GB | clip decode + 64-frame seek | ~1 s |
| KV cache @ 7.3k tokens | ~0.5 GB | vision encoder | ~0.4 s |
| linear-attn recurrent state | ~0.1 GB | LLM prefill (7.3k tokens) | ~3.2 s |
| vision + prefill activations | ~3 GB | decode ~20 output tokens | ~0.5 s |
| context / fragmentation | ~3 GB | | |
| **total** | **~61 GB / 80** | **total** | **~5 s / 15 s** |

Both fit with real margin. The KV cache is negligible because only **16 of 64
layers** use full attention, with 4 KV heads — a dense 27B would need several GB
here. Short outputs (bare answers) are what keep decode cheap.

*Verify in Phase 4:* the eval script times every sample and
`Evaluator.run(track=Track.SEGMENT)` enforces the 15 s ceiling by scoring
violations as **wrong**, not by erroring — so a latency regression looks like an
accuracy drop. Watch for it explicitly.

---

## 3. What was built

The phase order below is the order it was built in. Each step records the check that
proved it, because on this pipeline almost every failure is silent.

### Phase 0 — prerequisites

1. **Weights.** `Qwen/Qwen3.6-27B`, ~54 GB, cached locally; `preflight.py` §2 asserts
   the shards are present and >40 GB before a job is ever submitted.
2. **Frame coverage.** All 92 train + 38 test videos have complete `frames/`
   directories. `clip_sampling.frame_count()` establishes the length by binary search
   (~40 stat calls) rather than `os.listdir` over ~370k NFS entries.
3. **fps assertion, not assumption.** The builder reads each video's real fps with
   `decord` and aborts on any disagreement with `DATASET_BASE_FPS`. A 29.97-vs-30
   mismatch would drift ~3.6 s/hour — past the `time` tolerance, and silently. All
   videos are exactly 25.0 (heico) / 30.0 (lapchole); `--no-verify-fps` skips the
   check for re-exports.
4. **The anchor gate** moved into the base evaluation (§2.3).

### Phase 1 — dataset export: `build_segment_sft_dataset.py`

5. Per row: `sample_frame_indices()` gives N even, absolute indices spanning
   `[start_time, end_time]`, clipped to the frames on disk. Short clips get
   **repeated** indices, never a shortened list — constant sequence length keeps VRAM
   predictable; `n_distinct_frames` records the truth.
6. Records store `frame_dir` + `frames_indices`, **never expanded paths and never
   pixels**. 80 absolute paths per row would make the export ~10× larger for
   information `frame_file()` reconstructs exactly.
7. `videoID` and `uid` are **dataset-qualified** (`heico/…`). `qID` numbers from 1 in
   each dataset, so a handful collide across the merged export; the raw `qID` is kept
   because the Evaluator matches references on it per dataset.
8. **Video-level, procedure-stratified eval split** (`--eval-frac 0.10`, seed 42) via
   `make_eval_video_split()`, reused unchanged from the frame track. Segment clips are
   sliding windows over the same video asking the same question, so a row-level split
   would leak almost completely.

   Shipped export: **12,386 train / 1,360 eval** (83 / 9 videos) **/ 6,254 test**
   (38 videos), N=80 throughout. `tests/test_export.py` reconciles the counts against
   the source parquet and asserts all three splits are video-disjoint.

### Phase 2 — prompt

9. `../orena_sft/prompts.py` gained `track="segment"`: a `percentage` rule, the
   `<SECONDS seconds>` → `hh:mm:ss` conversion instruction with the explicit warning
   that markers count from the **start of the video, not the clip**, comma-separated
   multi-timestamps, and clip rather than frame wording. The frame arm is byte-identical
   to before, and `extract_answer()` is reused as-is.

### Phase 3 — training: `sft_train_qwen_segment_ddp.py`

10. **The collator** (`collate.py`) renders the full chat, finds the literal
    `<|im_start|>assistant\n<think>\n\n</think>\n\n` marker with `rindex`, and masks
    everything before it. `tests/test_collate.py` proves the tokenized prompt is an
    **exact token-level prefix** of the full sequence, that the supervised span decodes
    back to the bare answer, and — the check that matters most — that the rendered
    `<… seconds>` markers equal an **independently computed** `marker_times()`, are
    video-absolute, and bracket the ground-truth answer.
11. **Two processor contracts are enforced, not trusted** (§2.7): `do_sample_frames=False`
    on every call, and a `VideoMetadata` that `build_metadata()` refuses to construct
    with empty indices or non-positive fps.
12. **Config**: gradient checkpointing with `use_reentrant=False`; batch 1 × accum 16 ×
    2 GPUs = effective 32; LoRA r=8 α=16 dropout 0.05 on the 7 text projections with a
    frozen vision tower (39,845,888 params, 0.145%, confirmed by
    `print_trainable_parameters()`); lr 1e-4, bf16, seed 42; eval and save every 50
    steps. `ddp_find_unused_parameters=False` held — the flag exists but was never needed.
13. **In-training eval is subsetted and stratified.** The 1,360-clip eval split is 9
    videos, two of which contribute 400 clips each, and `percentage` has 3 clips in
    total — a random subset tracks two videos and usually loses a format entirely.
    `stratified_eval_subset()` round-robins over (video, format) groups and picks
    **once**, so the `eval_loss` curve carries no between-eval sampling noise.
14. `SampleGenerationCallback` generates and times one clip per answer format at every
    eval, `time` first, and flags anything over the 15 s ceiling. This is the tripwire
    for §2.3: a `time` prediction coming back as raw seconds or as a clip-relative
    offset means the timestamp plumbing is broken and no amount of training fixes it.
15. `preflight.py` gates the whole thing from CPU in seconds — kernels, weights, export
    ↔ `DEFAULT_N_FRAMES` agreement, split disjointness, `save_steps % eval_steps`,
    checkpoint disk space, prompt content.

### Phase 4 — evaluation: `evaluate_base_segment.py`

16. **This, not `evaluate_qwen_segment.py`, is the script every number came from.** It
    handles the base model and `--checkpoint-dir` alike, and adds what a 6,254-clip
    split at ~3 s each actually needs: `--shard i --num-shards n` interleaving (so every
    shard sees the same mix of clip durations), `--resume`, an untimed warmup call that
    absorbs the ~30 s fla/Triton JIT, and the frame prefetch of §2.6. `--mode score`
    merges the shards and runs the official `Evaluator` once with `track=Track.SEGMENT`.
17. Clips are built through `clip_sampling`/`collate` — the same modules training uses —
    and `--prompt-style` / `--frame-size` must match the training run.
18. `--judge-workers 1`, not 4: the judge is itself a Qwen3.5 hybrid, so judging goes
    through fla/Triton, whose autotuner is not thread-safe. Four workers crashed job
    12955 after a clean 4,000-row merge.
19. `build_stratified_subsample.py` produces `test_strat2000.jsonl` — 1,000 rows per
    procedure_type (equal, not proportional: the OOD and ID halves are read as separate
    numbers), proportional within a procedure over (format × clip length) with a floor
    of 1. Scoring a checkpoint drops from 1.26 h to ~24 min, which is what made the
    step-by-step sweep in §5 affordable.

### Phase 5 — packaging

20. `merge_lora.py` folds the adapter into the base weights and **verifies the merge**:
    an adapted projection must change and an unadapted tensor must not, or it refuses to
    save. A silent no-op merge would otherwise ship the base model. Output is a plain
    `Qwen3_5ForConditionalGeneration` (2 shards, 54.7 GB) that needs no peft at runtime,
    with `MERGE_INFO.json` and the source adapter config kept alongside for provenance.

---

## 4. Risks — how each one resolved

| risk | outcome |
|---|---|
| `time` answers stay unlearnable — anchors are N/2 | **Partly. Not for the reason predicted.** N=80 puts the anchor grid inside tolerance at every clip length (§2.3), yet `time` lands at 0.49 / 0.62 and `duration_estimation` at 0.29. The bottleneck is interpolation and conversion, not spacing |
| Marker → `hh:mm:ss` divmod slips eat the 2.3 s budget | **Real for the base model, trained away.** Base scores 0.21 / 0.18 on `time`; the same plumbing after SFT reaches 0.49 / 0.62. The overlay fallback was never built (§2.3) |
| 27B OOMs on H200 at batch 1 | **Did not happen** — 74 GB of 141 measured, ~10 GB under prediction at a *larger* N. Chunked cross-entropy never needed |
| Processor **re-samples** pre-sampled frames | **Contained** — `do_sample_frames=False` on every call in `collate.py`, asserted by `tests/test_collate.py` |
| Processor silently defaults to `fps=24` | **Contained** — `build_metadata()` raises on empty indices or non-positive fps; the test decodes `input_ids` and compares the rendered markers against an independent computation |
| Dataloader starves the GPU | **Did not happen** at 16 workers + `persistent_workers`. The repack to node-local shards was never needed. Inference needed its own prefetch (§2.6) |
| Overlapping sliding-window clips inflate the effective epoch | **Accepted, not mitigated.** The video-level split prevents leakage; the windows were left as augmentation and never deduplicated by (video, question, answer). Still a live question — it is part of why 1 epoch was enough |
| Only one GPU node, wall-clock limit | **Resolved by measurement** — 81.7 s/step × 650 = 14.8 h inside a 2-day request. Checkpoint-and-resume never exercised |
| `percentage` and rare capabilities are tiny | **As expected** — 42 train rows, 22/2 test rows, 0.32 / 0.50 accuracy. Noise; not tuned on |

**Settled:**
- **Model** — `Qwen3.6-27B`, LoRA r=8. The 9B control arm (`sft_train_segment_9b_rtx.slurm`)
  scores 5 points lower for a third of the parameters and half the latency — the 27B is
  worth it here only because the latency budget is not binding (§5).
- **Sampling** — flat **N=80** @ 640×360, even N, markers not overlay.
- **Hardware** — **2×H200** for the 27B, 2×RTX PRO 6000 for the 9B arm and for all
  evaluation. Inference at N=80 is ~61 GB, so evaluation never needs an H200.

**Still open, in priority order:**

1. **N=140 under SFT — the one decisive experiment not run.** The anchor-density model
   in §2.3 is the plan's central quantitative claim and it is unvalidated: the
   `sft_export_n140/` export was evaluated **base-only, 9B-only**, where `time` moved
   0.04 → 0.05. That is confounded — the base model is non-compliant, so the probe
   measures format-following, not grounding. With `time` at 38.5% of the data, still the
   worst large bucket, and ~4× latency headroom (9B at N=140 measured 2.08 s median), an
   SFT 9B run at N=140 either confirms anchor density is binding or retires the
   hypothesis and redirects the effort at decoding and prompt.
2. **`evaluate_qwen_segment.py` should be deleted.** `evaluate_base_segment.py` supersedes
   it entirely and produced every number on disk. Two near-identical `generate_one()`
   implementations on the one path where train/inference agreement matters is a standing
   drift risk.
3. **The test suite has drifted from the code.** `tests/test_collate.py` asserts N=64
   shapes (`T == 32`, 32 markers, ~110 tok/frame over 64) against an N=80 export, and
   `tests/test_train_config.py` asserts `eval_subset == 256` / `save_total_limit == 20`
   against defaults of 512 / `None`. These are the tests guarding the highest-risk module;
   red-for-stale-reasons is how they stop being run.
4. **`collate.py` runs the video processor twice per sample** — once on the full text and
   once on the prompt prefix, with the same 80-frame array — to locate the loss boundary.
   Since the answer is always the suffix, tokenizing `full_text[cut:]` and masking
   `labels[:-k] = -100` needs one call. BPE could merge across the boundary, so land it
   only with the exact-prefix assertion in `test_collate.py` §3 confirming it.
5. **`method_segment.md` and the journal entry were never written.** `journals/2026-07-29.md`
   covers day one only, and `journals/SUMMARY.md` still says "We use frame track only".
   Every result in §5 lives in CSVs and nowhere else.
6. **Merge with the frame track?** The two exports could train one multi-track adapter.
   Out of scope here; it would change the export layout.

---

## 5. The shipped run — `segment-27b-alldata-n80-650-20260730`

`checkpoints/segment-27b-alldata-n80-650-20260730/` (adapter, 159 MB) and
`…-merged/` (standalone weights, 2 shards, 54.7 GB). wandb run `o9px8d2w`, SLURM job
13109, `cgm-gpu38`, 2×H200, 2026-07-30 21:03 → 2026-07-31 11:49.

### Configuration

| | |
|---|---|
| base | `Qwen/Qwen3.6-27B` |
| adapter | LoRA r=8, α=16, dropout 0.05, `q,k,v,o,gate,up,down_proj` — 39,845,888 params (0.145%), vision tower frozen |
| clip | N=80 @ 640×360 → 40 anchors, 8,800 video tokens, 9,915-token prompt |
| prompt | `--prompt-style direct`, no FO definitions |
| batch | 1 × accum 16 × 2 GPUs = **32 effective** |
| optimizer | lr 1e-4 linear decay to 0, bf16, gradient checkpointing, seed 42 |
| **train file** | **`sft_export/all.jsonl` — 20,000 rows** |
| duration | `--max-steps 650` = **1.04 epochs**; early stopping disabled (`patience 999`) |
| result | train loss 0.173, eval loss 0.234, token accuracy 0.938 |

**`all.jsonl` is train ∪ eval ∪ test — the entire labelled segment track.** The public
test split ships with labels and the platform scores against its own hidden set, so
those rows are ordinary training data for a final model. Three consequences, all
deliberate:

- **Local evaluation of this checkpoint is meaningless.** Every test clip is in its
  training set. The numbers below come from `segment-27b-r8-direct-n80-20260729`, the
  honest-split sibling trained on the same recipe over `train.jsonl` only.
- **`eval_loss` is in-sample too** — `eval.jsonl` is a subset of `all.jsonl`. It tracks
  fit, not generalization, which is why early stopping was disabled and the run length
  fixed by `--max-steps` instead.
- **650 steps was chosen from the honest split, not from this run.** That split is
  12,386 rows = 387 steps/epoch, and the checkpoint sweep below plateaus at step ~400
  ≈ 1.03 epochs. At 625 steps/epoch on 20,000 rows, 650 steps is **1.04 epochs** — the
  same point on the curve, transferred by epoch fraction rather than by step count.

**Reproducibility gap:** `all.jsonl` has no builder flag — it is a concatenation of the
three exports, and nothing in this folder produces it. The frame track solved this
properly with `--include-test` on its builder (commit 73cc67b). Port that flag before
re-exporting, or the provenance of the shipped model's training data is a shell command
nobody wrote down.

### Results — honest split, full 6,254-clip test set

Same recipe over `train.jsonl` only, scored with the official `Evaluator`
(`track=Track.SEGMENT`, deterministic parsing for closed formats, LLM judge for the rest):

| arm | heico — Sigmoid Resection, **unseen procedure** | lapchole — same procedure, unseen videos | pooled |
|---|---|---|---|
| base Qwen3.6-27B | 0.324 | 0.361 | 0.337 |
| **SFT 27B** | **0.589** | **0.710** | **0.632** |
| SFT 9B, identical recipe | 0.550 | 0.641 | 0.583 |

By answer format (SFT 27B, heico / lapchole):

| | binary | fo_class | multiple_choice | time | number | open_ended | percentage |
|---|---|---|---|---|---|---|---|
| base | .62 / .67 | .29 / .23 | .55 / .50 | .21 / .18 | .28 / — | .67 / .69 | .09 / — |
| **SFT** | **.82 / .79** | **.70 / .74** | **.71 / .71** | **.49 / .62** | **.44 / —** | **.74 / .82** | **.32 / —** |

`—` marks cells with 2 test rows (lapchole `number`, `percentage`); heico
`percentage` has 22. Do not read those (§4).

Checkpoint sweep on `test_strat2000.jsonl` (1,000 rows per procedure, fixed seed):

| step | 50 | 100 | 150 | 200 | 250 | 300 | 400 | 450 | 500 | final |
|---|---|---|---|---|---|---|---|---|---|---|
| heico | .500 | .557 | .580 | .562 | .596 | .592 | **.617** | .601 | .603 | .597 |
| lapchole | .567 | .621 | .636 | .672 | .673 | .693 | .698 | .685 | .696 | **.707** |

### What these numbers say

- **SFT nearly doubles accuracy, and most of it is compliance.** `fo_class` .29 → .70
  and `time` .21 → .49 are not the model learning to see; they are the model learning to
  answer in the required shape. This replicates the frame track's central finding.
- **The OOD cost is real and SFT widened it.** heico test is Sigmoid Resection, a
  procedure in neither train nor eval. The base gap is 3.7 points; after SFT it is 12.1.
  Fine-tuning bought more on the in-procedure half.
- **`time` remains the weak spot** at 0.49 despite being 38.5% of the data, with
  `duration_estimation` at 0.29 and `temporal_localization` at 0.50. See §4, open item 1.
- **The 27B buys +5 points over the 9B** for 3× parameters and 2× latency. Defensible only
  because the 15 s ceiling is nowhere near binding.
- **Latency is a non-issue and was never exploited.** Median 2.89 s, max 4.32 s, zero
  violations. Roughly 4× headroom sat unused for the whole project — the single largest
  unspent resource in the pipeline, and the reason open item 1 is cheap to settle.
- **One caveat on those latencies:** they assume the frames are locally resident (§2.6).
  A submission container reading clips cold pays the prep cost the prefetch hides here.
  With 4× headroom this is comfortable, not free.
