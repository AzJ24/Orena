# Plan — SEGMENT-track SFT pipeline for Qwen3.6-27B

Port the frame-track method ([method_frame.md](../orena_sft/method_frame.md)) to the
**SEGMENT** track (video clips up to 5 min) with **`Qwen/Qwen3.6-27B`**. Same shape
as the frame pipeline — build a JSONL export, LoRA-SFT with TRL under DDP, evaluate
with the FOCUS `Evaluator` — but four things genuinely change: the visual input is a
*clip*, the question mix is dominated by **absolute-timestamp** answers, the model
is **3× larger**, and only **2 H200s** exist on this cluster.

All segment-track code lives in this folder. Shared, track-agnostic pieces
(`prompts.py`, `frame_path()`) stay in `../orena_sft/` as the single source of truth
and are imported via `sys.path.insert`, the pattern the repo already uses — see §2.5.

---

## 0. What is actually different from the frame track

| | FRAME (done) | SEGMENT (this plan) |
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

**Committed configuration:** Qwen3.6-27B · **N=64 frames** @ 640×360 · 2×H200 ·
LoRA r=8 · batch 1 × grad-accum 16 (effective 32) · gradient checkpointing ·
1.5 epochs. Rationale in §2.1, §2.3 and §2.8.

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

`Qwen/Qwen3.6-27B` — **only `config.json` is cached (32 KB); weights are not
downloaded.** The config declares:

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

## 2. Design decisions to make before writing code

### 2.1 Clip → frames: the sampling policy

Fixed **N frames uniformly spanning `[start_time, end_time]`**, not fixed fps.
Rationale: durations span 1 s → 300 s (300×), so fixed fps makes sequence length
vary by 300× and batching impossible; fixed N makes every sample the same token
cost, which is what lets `--batch-size` and the memory ceiling be predictable.

**Decision: flat N=64 @ 640×360.** Two constraints fix this, both measured (§2.7):

- **N must be even.** `temporal_patch_size=2` fuses frames in pairs; an odd N
  leaves a ragged pair.
- **Temporal resolution is N/2, not N.** The pairing means 64 frames give the model
  **32** time-anchors, not 64 — see §2.3 for what that costs.

Measured token cost (verified against the real processor, not estimated):

| frames | temporal positions | video tokens | tok/frame |
|---|---|---|---|
| 32 @ 640×360 | 16 | 3,520 | 110 |
| **64 @ 640×360** | **32** | **7,040** | **110** |

`--num-frames` and `--frame-size` stay CLI flags, but N=64 is the committed
default: accuracy on the `time` bucket is the priority, and §2.8 shows N=64 fits
2×H200 with headroom and stays well inside the challenge's inference budget.

*Optional refinement, not adopted:* an adaptive `N = min(2·ceil(duration/4), 64)`
would cut short clips' cost with no accuracy loss (a 29 s clip needs nowhere near
32 anchors). It saves **time, not memory** — peak VRAM is still set by the 300 s
clips that hit the cap — and it makes step time variable. Revisit only if the
wall-clock becomes binding.

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

| N | anchors | spacing on a 299 s clip | tolerance | hit rate |
|---|---|---|---|---|
| 32 | 16 | 19.3 s | ±4.3 s | ~45% |
| **64** | **32** | **9.6 s** | ±4.3 s | **~90%** |

On the median 119 s clip N=64 gives 3.8 s spacing against a ±2.3 s tolerance —
covered outright. **This is why §2.1 commits to N=64**; N=32 would leave over half
the largest answer bucket unanswerable by construction. The anchor mechanism below
is second-order to this.

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
2.3 s budget gets blown by an off-by-one-minute slip. **That is settled empirically
by the Phase-0 probe (§3, step 2), before any export or training exists.** If the
base model converts cleanly → markers only, and the overlay drops out of the plan.
If it fumbles the arithmetic → build the overlay arm too.

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

```
segment_track/
  plan.md                          ← this file
  build_segment_sft_dataset.py     ← Phase 1
  sft_train_qwen_segment_ddp.py    ← Phase 3
  sft_train_qwen_segment_ddp.slurm
  evaluate_qwen_segment.py         ← Phase 4
  probe_timestamp_markers.py       ← Phase 0 gate
  sft_export/{train,eval,test}.jsonl
  checkpoints/<run_name>/
  logs/
  method_segment.md                ← Phase 5
```

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
lapchole 2.0) — one export only touches ≈35 GB at N=32 (13.7k rows × 32 × ~80 KB).
It is wrong because copying ~440k small files *is itself* ~440k round-trips:
≈4.9 h single-threaded to stage that 35 GB. The data is fine where it is; what has
to change is how many reads are in flight.

1. **Parallel dataloader workers — the actual fix.** The latency is per-file and
   embarrassingly parallel. One optimizer step is 16 micro-batches × 32 frames =
   512 frames per GPU: 36 s single-threaded, ~1.1 s across the 32 CPUs the SLURM
   script already requests. A 27B fwd+bwd at ~3.5k tokens × 16 accumulation steps
   is on the order of 10–20 s, so loading hides completely behind compute. Set
   `dataloader_num_workers` to 16–24 with `persistent_workers=True`.
2. **The page cache stages itself, for free.** ~35 GB working set against a 256 GB
   `--mem` request, so after the first epoch every frame is resident and reads drop
   to 9 ms — no code, no copying. Optional pre-warm: the export knows every path up
   front, so a parallel `cat` pass over the file list warms it in ~10 min if the
   first epoch proves too slow.
3. **Only if the smoke run shows starvation:** repack at export time into one file
   per clip on node-local scratch, pre-resized to the training resolution (35 GB →
   ~15 GB at 640×360). That turns 32 round-trips per sample into one sequential
   read and drops the decode cost too. Real engineering — justify it with a
   measured profile, not an assumption.

The decision point is the Phase-3 smoke run (§3, step 8): log GPU step time against
dataloader wait time there and it will be obvious whether 1 and 2 suffice.

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

### 2.8 Budgets — training and inference

**Training, per GPU, N=64 (~7.3k tokens), batch 1, LoRA r=8 + gradient
checkpointing:**

| component | size |
|---|---|
| bf16 weights (~27B) | 54 GB |
| LoRA params + AdamW states + grads | ~1 GB |
| activations (checkpointed) | ~16–20 GB |
| logits (7.3k × 248,320 vocab) | ~4–8 GB |
| CUDA context, fragmentation | ~4 GB |
| **total** | **~80–85 GB** |

Fits 2×H200 (141 GB) with ~55 GB headroom. **Would be marginal on the 96 GB
RTX PRO 6000s** — which is why the H200 path is the committed config. The 248k
vocabulary makes the logits tensor unusually large; if memory gets tight, a chunked
cross-entropy is the first lever, before touching N.

**Wall clock**, 430 optimizer steps/epoch (13,746 ÷ 32), ±2× until the smoke run:

| | per step | 1 epoch | **1.5 epochs** | 2 epochs |
|---|---|---|---|---|
| 2×H200, N=64 | ~60–80 s | ~8 h | **~12–13 h** | ~16–17 h |

1.5 epochs fits the 23 h SLURM limit; 2 epochs needs checkpoint-and-resume.

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

## 3. Implementation steps

### Phase 0 — prerequisites and the anchor gate

1. **Start the weight download** (slow, run it in the background first).
   `HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3.6-27B` (~54 GB into
   `~/.cache/huggingface`, 347 GB free).
   *Verify:* `AutoProcessor.from_pretrained` + `Qwen3_5ForConditionalGeneration.
   from_pretrained(..., dtype=bfloat16)` load on gpu38 and a single-frame
   `generate()` returns text.

2. **`probe_timestamp_markers.py` — decide the §2.3 anchor before building
   anything.** Needs no export, no training script, and no download: use the
   **already-cached `Qwen3.5-27B`** (52 GB on disk, identical processor and model
   class). ~1 h of GPU time.
   - take ~50 `time`-format segment rows spanning short and long clips
   - sample N frames, build `VideoMetadata(fps=base_fps,
     frames_indices=<absolute>, total_num_frames=…)`, prompt for a bare `hh:mm:ss`
   - measure two things separately: **(i) does it use the markers at all** — does
     the answer track the marker values, or is it clip-relative / anchored to the
     timestamp quoted in the question text; **(ii) is the divmod correct** — is
     `|pred − nearest marker|` consistent with the marker it should have read, and
     how often is the error a clean 60 s multiple (the arithmetic-slip signature)?
   - run the same 50 against `frames_overlay/` frames for a direct comparison
   *Gate:* markers convert cleanly → **markers only**, delete the overlay arm from
   Phase 1. Arithmetic slips dominate → build both arms and ablate.
   *Bonus, free:* this also yields the real measured token count per frame at each
   candidate resolution, replacing the estimates in §2.1.

3. **Confirm frame coverage for the segment videos.** For all 92 train + 38 test
   videos, check `frames/<stem>/` exists and its max frame index ≥
   `round(max end_time · base_fps)`; assert each video's actual fps equals
   `DATASET_BASE_FPS` (§2.3). Any gap means re-running
   `FrameExtractorPreprocessor` (slow) — find out now, not at hour 3 of training.
   *Expected:* full coverage (§1.2).

### Phase 1 — dataset export: `build_segment_sft_dataset.py`

Mirror [build_frame_sft_dataset.py](../orena_sft/build_frame_sft_dataset.py);
import `frame_path()` from it rather than re-deriving the formula (§2.5).

4. **Write the builder.**
   - `FocusDataset(dataset, DatasetSplit.TRAIN|TEST, Track.SEGMENT)`
   - per row: compute `frames_indices = uniform N indices in [round(start·fps),
     round(end·fps)]` with **N=64 and N always even** (§2.1), clipped to the frames
     actually on disk
   - short clips with fewer than N distinct frames (a 1 s clip has only 25) get
     **repeated indices**, not a shortened list — constant sequence length is worth
     more than the wasted tokens, and it keeps VRAM predictable. Record the true
     distinct-frame count in the export
   - record fields: frame track's metadata **plus** `start_time`, `end_time`,
     `duration`, `base_fps`, `frames_indices`, `frame_paths`, `ood`, `generation`
   - `messages`: `{"type": "video", "video": [<frame paths>]}` + question / bare
     answer — same two-turn chat shape as the frame export
   - **video-level, procedure-stratified eval split** (`--eval-frac 0.12`,
     `seed 42`), reusing `make_eval_video_split()` unchanged
   - store **frame paths, never pixels** (the frame track's rule; here it matters
     N× more)
   *Decision to record:* `--num-frames` is baked into the export, so a sampling
   ablation means re-exporting. That is cheap (KB-scale JSONL, no decoding) and
   keeps the training script simple. Write one export per N.
5. **Run and sanity-check it.**
   `.venv/bin/python segment_track/build_segment_sft_dataset.py --datasets heico
   lapchole --num-frames 64 --out-dir segment_track/sft_export`
   *Verify:* ~13.7k train rows before the split, no video appears in both train and
   eval, every `frame_paths` entry exists on disk, `frames_indices` are monotone
   and inside the window, format/capability distributions match §1.1.

### Phase 2 — prompt

6. **Extend `../orena_sft/prompts.py`** per §2.4 (`track="segment"` arm:
   percentage rule, `hh:mm:ss`-from-seconds-markers instruction, multi-timestamp,
   clip wording).
   *Verify:* `build_system_prompt(track="frame")` output is **byte-identical** to
   today's (regression-guard the frame runs), and `extract_answer` unit cases still
   pass for `hh:mm:ss` and `12%`.

### Phase 3 — training: `sft_train_qwen_segment_ddp.py`

Fork [sft_train_qwen_frame_ddp.py](../orena_sft/sft_train_qwen_frame_ddp.py) — the
DDP scaffolding (rank-guarded generation callback and preview, `WANDB_MODE` guard,
`device_map={"": LOCAL_RANK}`, `ddp_find_unused_parameters=False`, wandb config
snapshot, `processing_class=processor`) transfers unchanged and is all hard-won;
only the collator and the model size change.

7. **Rewrite the collator for video** (§2.2/§2.3/§2.7):
   load N JPEGs → resize → `processor(text=…, videos=[frames],
   video_metadata=[VideoMetadata(fps=base_fps, frames_indices=<absolute>,
   total_num_frames=<source video length>)], **do_sample_frames=False**)` → pad
   `input_ids` / `attention_mask` / `labels` / `mm_token_type_ids`, concat
   `pixel_values_videos` and `video_grid_thw`. Keep the marker-slice loss masking
   exactly as-is.
   *Verify, with a standalone script before any training — all of it runs on CPU:*
   the assistant marker occurs exactly once; the prompt tokenization is an exact
   prefix of the full tokenization; the unmasked target decodes back to the bare
   answer; and **decoding `input_ids`** (not the pre-processor chat-template
   string — the markers are injected inside `processor.__call__`) yields N/2
   `<… seconds>` markers whose values equal the mean of each fused pair's
   `frames_indices / base_fps`. §2.7 is a worked example of exactly this check;
   reuse its assertions.
8. **Adjust the training config for 27B on 2×H200:**
   - `gradient_checkpointing=True` (+ `gradient_checkpointing_kwargs=
     {"use_reentrant": False}`) — **required** at N=64 (§2.8); note it may force
     `ddp_find_unused_parameters=True` with LoRA, per
     [ddp.md](../orena_sft/ddp.md) §3c
   - `--batch-size 1 --grad-accum 16` → **effective batch 32**, matching the frame
     runs so the 1.5-epoch convergence estimate carries over
   - LoRA `r=8, alpha=16` on the same 7 projections (frozen vision tower) as a
     starting point; `r=16` is the first thing to try if it underfits
   - keep `lr 1e-4`, bf16, `seed 42`, eval/save every 150 steps
   - freeze the vision tower explicitly and confirm via
     `print_trainable_parameters()`
   *Verify:* `--max-steps 20` smoke run completes, peak VRAM is logged (expect
   ~80–85 GB of 141), and step time is recorded → extrapolate against the 23 h
   SLURM limit **before** launching (§2.8 predicts ~12–13 h for 1.5 epochs). Log
   **GPU step time against dataloader wait time** in the same run — that is the
   §2.6 decision point. If VRAM does not fit, the escalation order is: chunked
   cross-entropy (the logits tensor is the soft target) → smaller frames → lower N
   → `r`/target-module reduction → 4-bit QLoRA.
9. **Write `sft_train_qwen_segment_ddp.slurm`**: same as the frame DDP wrapper
   (`gpu:h200:2`, `torchrun --standalone --nproc_per_node=2`, `NCCL_IB_DISABLE=1`,
   `TOKENIZERS_PARALLELISM=false`), keeping `--cpus-per-task=32` — those CPUs are
   what makes the frame I/O disappear (§2.6). Set `dataloader_num_workers` 16–24
   and `persistent_workers=True`.
10. **Launch the real run** (wandb project `orena-segment-sft`) and watch the
    rank-0 `eval_samples.jsonl` for `time` answers specifically — if they come out
    as `<…> seconds` or as clip-relative offsets, §2.3 is broken and no amount of
    training will fix it.

### Phase 4 — evaluation: `evaluate_qwen_segment.py`

11. **Fork [evaluate_qwen_frame.py](../orena_sft/evaluate_qwen_frame.py)**: same
    `focus.evaluation.Evaluator` path (deterministic parsing for binary/number/
    percentage/fo_class/time, LLM-judge for open_ended/matching/multiple_choice),
    but build clips with the Phase-1 sampler and call
    `Evaluator.run(..., track=Track.SEGMENT)` so the **15 s** latency ceiling
    applies. Keep left-padded batched generation and per-sample amortized timing.
    *Non-negotiable:* evaluate with the **same** `--num-frames`, `--frame-size`,
    prompt arm, and `--condition-procedure` as training, or the numbers are a
    train/inference mismatch.
12. **Baseline first, then compare.** Score base `Qwen3.6-27B` (prompt-only,
    `plain` and `direct`) on the segment test split *before* reading any SFT
    number — the frame-track finding was that base VLMs are non-compliant rather
    than blind, and the same 2×2 (SFT × prompt) is what makes the result
    interpretable. Break accuracy out **by `answer_format` and by capability
    group**; report `time` separately, since at 38.5% of the data it will dominate
    the headline number and behaves unlike everything else.
13. **Report the OOD split honestly:** heico test is Sigmoid Resection, a
    procedure in neither train nor eval — that is the real generalization number.

### Phase 5 — write-up

14. **`method_segment.md`** in this folder, same structure as
    [method_frame.md](../orena_sft/method_frame.md) (data / model / method), plus a
    `journals/` entry recording the probe result, the measured token counts, VRAM,
    step time, and the N-frames ablation.

---

## 4. Risks and open questions

| risk | severity | mitigation |
|---|---|---|
| `time` answers stay unlearnable — anchors are N/2, so even N=64 gives only 9.6 s spacing on a 300 s clip (~90%) | **high** — 38.5% of the data | N=64 committed (§2.1); the longest clips remain the weak point — report `time` accuracy split by clip duration to see it |
| Marker → `hh:mm:ss` divmod slips (off-by-one-minute) eat the 2.3 s budget | high | **Phase-0 probe (step 2) settles this before anything is built**; overlay arm as the fallback |
| 27B at N=64 OOMs on H200 at batch 1 | medium — §2.8 predicts ~80–85 GB of 141 | chunked cross-entropy first (248k-vocab logits are the largest soft target), then frame size, then N |
| Processor **re-samples** pre-sampled frames | high | `do_sample_frames=False` — mandatory, see §2.7; crashes loudly in one config and silently corrupts timestamps in another |
| Processor silently defaults to `fps=24` | high, **silent** | assert `video_metadata` populated; unit-test the `<… seconds>` values decoded from `input_ids` (§2.7) |
| Overlay draw params drift between training and submission | medium, **silent** | pin font/scale/thickness/position in one shared constant; only a risk if the overlay arm is built |
| Dataloader (N JPEG reads/sample over NFS) starves the GPU | medium — quantified in §2.6 | cold NFS is 71 ms/frame (latency-bound), so 512 frames/step is 36 s single-threaded but ~1.1 s across 32 CPUs; `dataloader_num_workers` 16–24 + `persistent_workers`, then the page cache (35 GB working set vs 256 GB `--mem`). Repack to node-local per-clip shards only if the smoke run shows starvation |
| Overlapping sliding-window clips inflate the effective epoch | medium | video-level split (already); consider deduplicating by (video, question, answer) and treating windows as augmentation |
| Only one GPU node in the cluster, 23 h limit | medium | measure step time in the smoke run and size `--epochs` to fit; `--save-steps` + resume-from-checkpoint |
| `percentage` and the rarer capabilities are tiny (42 rows) | low | expect noise; do not tune on them |

**Settled:**
- **Model** — `Qwen3.6-27B`. Weights downloading.
- **Sampling** — flat **N=64** @ 640×360, even N. Accuracy on the `time` bucket was
  chosen over training cost, with §2.8 confirming it fits.
- **Hardware** — **2×H200**, not the 4×RTX PRO 6000: at N=64 the ~80–85 GB working
  set is comfortable on 141 GB and marginal on 96 GB.

**Still open:**
- **Merge with the frame track?** The two exports could train one multi-track
  adapter. Out of scope here, but it changes the export layout if wanted early.
- **1.5 vs 2 epochs** — 1.5 fits the 23 h limit outright; 2 needs
  checkpoint-and-resume. Decide after the smoke run gives a real step time.
