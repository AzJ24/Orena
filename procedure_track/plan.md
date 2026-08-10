# Plan — PROCEDURE-track SFT pipeline for Qwen3.5-9B

Port the SEGMENT-track pipeline ([../segment_track/plan.md](../segment_track/plan.md))
to the **PROCEDURE** track: the visual context is a *prefix of a whole operation*,
median 1 h 40 m and up to **4 h 56 m**. The schema is identical to segment, so the
export → collate → train → evaluate machinery transfers with almost no change. What
does not transfer is the arithmetic: at this scale uniform frame sampling cannot
reach the `time` tolerance, and no choice of N fixes that.

This is a **forward-looking plan**. §1 is measured — every number is read off the
parquet or off disk, and marked when it is not. §2 estimates are extrapolated from
the segment track's *measured* 74 GB / 81.7 s-per-step / 2.89 s-inference numbers and
are labelled as estimates until the smoke run replaces them.

**Committed configuration for the first submission:** **Qwen3.5-9B** · LoRA r=8 α=16 ·
**capped-5s + question-anchored sampling, N_max=768** @ 640×360 · **`overlayed/` video**
· ~1 epoch over all 10,000 rows. Rationale in §2.2–§2.4.

Three of those were different when this file was first written — 27B, flat N=256, and
plain frames. All three moved after the organizers published the submission template
and the GPU tiers (§1.6), and after the Phase-0 checks were actually run (§3). The
reasons are recorded in place; the short version is that the input is now a 5 fps
clip with a clock already burned into it, and that at equal training cost a smaller
model buys 3× the temporal resolution on the one axis this track is hard along.

---

## 0. What is actually different from the segment track

| | SEGMENT (shipped) | PROCEDURE (this plan) |
|---|---|---|
| visual input | clip `[start, end]`, median 119 s | **prefix `[0, t]`, median 1 h 40 m** |
| longest input | 300 s | **17,780 s (4 h 56 m)** |
| `time` tolerance | 2.3 s of 119 s — 1 in 52 | **5.0 s of 2,950 s — 1 in 589** |
| dominant format | `time` 38.5% | `time` 36.1%, but `number` triples to 21.5% |
| rows | 20,000 | **10,000** (6,873 train / 3,127 test) |
| latency budget | 15 s/question | **30 s/question**, pooled as `120 s + B × 30 s` |
| inference VRAM | 80 GB assumed | **96 GB** confirmed (RTX PRO 6000 Blackwell) |
| model | Qwen3.6-27B | **Qwen3.5-9B** — §2.2 |
| frames per sample | flat 80 (8,800 tokens) | **capped-5s, N_max 768** (88,320 tokens) — §2.2 |
| timestamp mechanism | text markers only | **markers + burned-in clock** — §2.3a |

**The single hardest thing** is no longer "can the model read a timestamp marker" — it
is that **the markers cannot be dense enough**. §1.3 shows the required context exceeds
the model's position budget by 1.5×. That is a wall, not a tuning problem, and the plan
is built around routing past it rather than through it.

---

## 1. Facts established (read from the data, not assumed)

The procedure parquets are not part of the default cache — `data/procedure/{train,test}.parquet`
must be fetched explicitly from `orena-dkfz/{heico,lapchole}-focus-vqa`.

### 1.1 Data

| dataset / split | rows | videos | procedure |
|---|---|---|---|
| heico train | 4,000 | 20 | Proctocolectomy (2,000) + Rectal Resection (2,000) |
| heico test | 2,000 | 10 | **Sigmoid Resection** (unseen procedure) |
| lapchole train | 2,873 | 72 | Lap. Cholecystectomy |
| lapchole test | 1,127 | 28 | Lap. Cholecystectomy |
| **combined train** | **6,873** | **92** | |
| **combined test** | **3,127** | **38** | |

Rows per video: min 25, median 40, max 200. Same 130 videos as the segment track, and
the same OOD structure — heico test is a procedure type in neither train split.

Answer formats, combined train: `time` 2,481 (36.1%) · `fo_class` 1,723 (25.1%) ·
`number` 1,478 (21.5%) · `binary` 562 (8.2%) · `open_ended` 440 (6.4%) ·
`multiple_choice` 144 (2.1%) · `percentage` 45 (0.7%).

Against the tracks already built (train rows, combined):

| format | frame | segment | **procedure** |
|---|---|---|---|
| `time` | 0 | 5,285 | 2,481 |
| `fo_class` | 6,296 | 3,441 | 1,723 |
| `number` | 4,261 | 896 | **1,478** |
| `multiple_choice` | 640 | 2,186 | **144** |

Two shifts matter: `number` is **3× more prominent** than in segment and the answers are
much larger (mean 10.9, median 4, max 76 — counting over hours, not minutes), while
`multiple_choice` all but disappears, taking most of the LLM-judged share with it.
`fo_class` answers are multi-class 32.1% of the time; `none` is the answer 292 times.

**The row schema is identical to frame and segment** (`id, video, procedure_type,
question, answer, answer_format, track, generation, clinical_relevance, ood,
timestamp_start, timestamp_end, primary_capability, secondary_capabilities`), so
`FocusDataset(dataset, split, Track.PROCEDURE)` parses it unchanged. `ood` is `False`
everywhere in the released data; `clinical_relevance` is `True` on 49 of 10,000 rows.

Primary capability is dominated by temporal grounding and aggregation — `2a`
temporal_localization 3,387 · `1a` object_identification 2,202 · `3a` object_aggregation
1,649 · `3b` event_aggregation 1,082. All five groups appear, but complex reasoning
(`5a`/`5b`/`5c`) totals 219 rows: report it, do not tune on it.

### 1.2 The window is a prefix, not a clip

`timestamp_start` is `00:00:00` in **100%** of rows. Each video appears under a median
of **33 distinct end times** (min 16, max 185), so a procedure sample is *"the video up
to time t"*, growing.

This is load-bearing, not cosmetic. Of the 1,478 (video, question) pairs that recur
across more than one window, **46.5% change their answer** with the end time. The
boundary is a model input — "how many clips have been inserted" genuinely differs at
40 minutes and at 3 hours. §2.4 is the consequence.

| window duration | heico | lapchole |
|---|---|---|
| min | 309 s | 309 s |
| median | **5,989 s (1 h 40 m)** | 1,579 s (26 m) |
| p95 | 14,040 s (3 h 54 m) | 3,729 s (1 h 02 m) |
| max | **17,780 s (4 h 56 m)** | 5,135 s (1 h 26 m) |

Source video lengths: heico median 3 h 09 m, max 4 h 56 m; lapchole median 33 m. The
*shortest* procedure window is longer than the *longest* segment clip — the two tracks
do not overlap in scale at all.

### 1.3 The tolerance wall

`time` tolerance is `min(5.0, 1 + duration·(4/360))`
(`focus/data/base_dataset.py:180`). The shortest window is 309 s → 4.43 s; anything
above ~450 s saturates at **5.0 s**. Every procedure `time` question is therefore
effectively at the cap.

| track | tolerance | window | precision |
|---|---|---|---|
| SEGMENT (median) | 2.3 s | 119 s | 1 in 52 |
| **PROCEDURE (median)** | **5.0 s** | **2,950 s** | **1 in 589** |
| PROCEDURE (worst) | 5.0 s | 17,780 s | 1 in 3,556 |

Qwen3-VL emits one timestamp anchor per *fused frame pair*, so N frames give N/2
anchors at a measured 110 tokens/frame (§2.2 of the segment plan, verified against the
real processor):

| N | video tokens | spacing on a 4 h 56 m video | spacing on median heico (3 h 09 m) |
|---|---|---|---|
| 80 (segment shipped) | 8,800 | 444 s | 283 s |
| **256** | **28,160** | **139 s** | **89 s** |
| 640 | 70,400 | 56 s | 35 s |
| 2,048 | 225,280 | 17 s | 11 s |

Landing inside ±5 s needs spacing ≤ 10 s, i.e. `N/2 ≥ 1,778` → **N ≈ 3,556 frames ≈
391,000 video tokens** for the longest window. Qwen3.5-9B and Qwen3.6-27B both declare
`max_position_embeddings: 262,144` (checked). **The required context does not fit either
model**, before memory or the 30 s budget are even considered — and 7.7% of rows exceed
that limit outright. §2.2 caps the policy accordingly.

This is the sharpest difference from segment. There, N=80 satisfied the anchor-spacing
condition and the model still reached only 0.49 — the necessary condition was met and
something else bound. Here the necessary condition is **unreachable by construction**,
so a straight port cannot work on 36% of the data. §2.3 is the cheap partial answer;
anything better is a second-submission problem (§4).

`time` answers are comma-separated multi-timestamp in 3.1% of rows, up to **20
timestamps in one answer**. Single answers sit late in the window — mean 0.625 of
elapsed time, median 0.686, p95 0.993. That skew is exploitable prior structure and is
probably where a nontrivial share of any `time` accuracy will come from; do not mistake
it for grounding.

### 1.4 The questions are genuinely long-horizon

2,724 distinct questions over 574 templates. The largest templates are not segment
questions on longer clips:

- *"At what time was a Clip first visible in the video?"* / *"last visible"* — 672 rows
- *"There is one Sponge in the frame at `<T>`. When is it retrieved from the surgical
  site?"* — persistent tracking across hours
- *"How many separate times does a Sponge completely leave the field of view for `<D>`
  frames or more and then return?"* — 615 rows
- *"What is the maximum number of Clips appearing at once in a single frame?"* — a max
  over **every** frame; the longest video is 444,500 frames at 25 fps
- *"Of the foreign objects inserted in the surgical site, how many were retrieved
  again?"* — the retained-foreign-object question the challenge exists for

**27.8% of questions quote at least one timestamp** (2,781 rows) — the handhold §2.3 is
built on. Generation: `automatic` 5,822 · `anchor` 3,706 · `manual` 472. A further 704
rows (*"What types of foreign objects are seen between `<T>` and `<T>`?"*, median
sub-window 541 s) are effectively segment-track questions wearing a procedure label.

### 1.5 Frames on disk for training; 5 fps MP4 at inference

**Training reads JPEGs.** All **130/130** procedure videos have complete extractions
under `/projects/datasets_ML/orena/{heico,lapchole}/frames/<stem>/` and
`frames_overlay/<stem>/` — verified every directory exists and its max index covers the
full video length. **15,464,740 frames, ≈1.24 TB.** No extraction work needed, but the
working set is ~10× the segment track's, so staging is impossible and the page cache
will not hold it (§2.7).

Native resolutions are **not** uniform, contrary to what the segment plan assumed:
heico is 960×540 throughout, lapchole spans 1280×720 (77 videos), 720×576 (6),
720×480 (6), 854×480 (5), 640×360 (5), 640×480 (1) — four aspect ratios. Everything is
resized to 640×360 regardless, in training and at inference alike, so the geometry
stays consistent; it matters only for §2.3a.

**Inference reads video.** Nothing about the JPEG layout survives into the container —
see §1.6.

### 1.6 The submission contract (from the released template)

`IMSY-DKFZ/orena-focus-submission-template`, quoted verbatim from its README:

> "Video clips are H.264 MP4 at exactly **5 fps** (one frame every 0.2 s),
> height-normalised to at most 576 px; widths follow each source's aspect ratio
> (e.g. 1024×576, 720×576 or 640×360), so **do not assume a fixed resolution**. Clips
> carry no audio, and **a keyframe every 5 s keeps seeking cheap**. The overlay clock
> shows the source timeline's second (floored)."

What follows from it:

- **Two variants per question**: `plain/<qID>.mp4` and `overlayed/<qID>.mp4`, the latter
  with `HH:MM:SS` burned into every frame. The organizers now supply the clock we would
  have had to burn ourselves — which reverses the segment track's overlay verdict
  (§2.3a).
- **The clip is pre-trimmed and must be decoded from its own beginning** — *"do not seek
  to `start_time` inside it."* `start_time`/`end_time` refer to the original procedure
  timeline. For PROCEDURE `start_time` is always `00:00:00` (§1.2), so the clip's frame 0
  *is* operation time 0; the index-origin correction the segment track needs does not
  apply here. **The fps correction does** — see §2.1.
- **Seeking is cheap**: a keyframe every 5 s at 5 fps is one every 25 frames, so 768
  scattered seeks decode ~19,200 frames rather than the 90,000 of a sequential pass.
  Seek; do not stream. This retires the plan's original worst risk (§2.8).
- **Latency is pooled**, `120 s + B × 30 s`, not a hard per-question cutoff. Model load
  comes out of the 120 s grace. The Evaluator still scores an over-budget response as
  **wrong** rather than erroring, so a latency regression looks like an accuracy drop.
- I/O is `request.json` (a list of `focus.Request`) → `answer.json` (a list of
  `focus.Response` with `qID`, `content`, `latency`).

**Hardware:** **96 GB** at inference (RTX PRO 6000 Blackwell), confirmed. The 48 GB L40S
tier is the alternative and would not hold a 27B at all. Training stays 2×H200 (141 GB)
with a 2-day SLURM limit. Note the 96 GB tier queues **12 h–3 days**, against a
**September 1** deadline and 10–11 submissions per track — so iteration count, not
accuracy per run, is the scarce resource.

---

## 2. Design decisions

### 2.1 Reuse the segment code; fork nothing

`clip_sampling.py`, `collate.py`, the trainer, `merge_lora.py` and `preflight.py` are
track-agnostic once N is a flag. The schema is identical (§1.1) and the processor
contracts are the same, so the two things that took longest to get right in the segment
track — `do_sample_frames=False` and an asserted `VideoMetadata` — carry over for free.

**Import them, do not copy them.** The segment track's own rule, and here it matters
more: two samplers that disagree about which frames a `[0, t]` window resolves to is a
silent train/inference mismatch that no test would catch after the fact.

The one genuinely new piece is the sampler policy (§2.3). Add it as a function inside
`clip_sampling.py` behind a flag, so both tracks keep one code path.

**Everything works in the 5 fps index space.** Training reads native-fps JPEGs and
inference reads a 5 fps MP4, and the only safe way to keep them agreeing is to make
5 fps the single canonical timeline and convert at the edges:

| where | conversion |
|---|---|
| sampler | returns `idx5`, absolute indices on the 5 fps grid |
| export (find the JPEG) | `native_idx = round(idx5 · base_fps / 5)` |
| inference (seek in the clip) | `local_idx = idx5 − round(start_time · 5)`, which is `idx5` for procedure |
| **metadata, always** | `VideoMetadata(fps=5.0, frames_indices=idx5)` |

heico at 25 fps: `idx5=2775` → native 13875 → marker `2775/5 = 555.0 s`. lapchole at
30 fps: `idx5=2775` → native 16650 → the same 555.0 s. One number, one meaning, both
paths.

**This is verified, not assumed.** `verify_fps_roundtrip.py` encodes a real training
window to a genuine 5 fps MP4, decodes it back, runs the real processor, and asserts the
rendered markers match the training path — they agree exactly (565.0 vs 565.0). It also
exercises both ways of getting it wrong, because both fail silently:

- keeping the training `fps` (25/30) with 5 fps indices renders every timestamp
  **5× too early** — 113.0 s where 565.0 s is correct;
- omitting the origin shift costs the full window offset on segment (10.0 s vs 565.0 s)
  and is **harmless on procedure**, where `start_time` is always 0.

Run it before any submission. A wrong marker is invisible in local evaluation, which
reads the export rather than a video.

### 2.2 Qwen3.5-9B, capped-5s sampling at N_max=768

**Sampling policy: one frame every 5 s, capped at N_max; longer windows stretch to a
uniform N_max grid.**

5 s is not arbitrary. Frames fuse in pairs, so one frame per 5 s gives one anchor per
**10 s**, against a ±5 s tolerance — a 10-second window. That is exactly the density at
which a `time` answer becomes reachable, and the coarsest that qualifies. It does not
fit universally (§1.3), so cap it and degrade gracefully.

| N_max | full-5s windows | rows at **true** 5 s density | fallback at median heico | tokens |
|---|---|---|---|---|
| 256 | ≤ 21 min | 18.0% (heico 6.2%, lap 35.8%) | 46.8 s/anchor | 29 k |
| 512 | ≤ 43 min | 42.5% (heico 18.3%, lap 78.8%) | 23.4 s/anchor | 59 k |
| **768** | **≤ 64 min** | **56.6%** (heico 30.4%, lap **95.8%**) | **15.6 s/anchor** | **88 k** |
| 1024 | ≤ 85 min | 64.8% (heico 41.4%, lap 99.9%) | 11.7 s/anchor | 118 k |

**N_max=768** takes essentially all of lapchole to true 5-second density and still fits
both training runs. Actual cost will beat the table: 56.6% of rows use *fewer* than 768
frames, so mean sequence length is well below the cap.

**Model: Qwen3.5-9B, not the 27B.** With 96 GB confirmed the 27B fits at inference, so
this is a trade rather than a constraint — and the trade is decided by training cost.
At an equal ~23 h budget:

| | rows at true 5 s density | fallback, median heico |
|---|---|---|
| 27B @ N=256 | 18.3% | 46.8 s/anchor |
| **9B @ N=768** | **56.6%** | **15.6 s/anchor** |

A 27B at N=512 would need ~48 h for one epoch and does not fit the wall at all. The
27B's measured +5 points on segment came from a track where coverage was never the
binding constraint; here it is, and 3× the temporal resolution is the better buy. The
12 h–3 day queue on the 96 GB tier (§1.6) reinforces it — fewer shots means each should
be the configuration one actually believes in.

**Estimates**, extrapolated from segment's measured 74 GB peak, 81.7 s/step at 8,800
tokens on the 27B, and 2.89 s median inference:

| | 9B @ N_max 768 (88 k tokens) |
|---|---|
| inference, of 30 s | ~18–20 s |
| training | ~270 s/step |
| honest-split run (215 steps) | ~16 h |
| all-data run (313 steps) | ~23 h |

Both fit the 2-day wall as separate jobs, ~40 h total across a 22-day calendar.
**Fallback is N_max=512** (~11 h + ~16 h) if the smoke run's step time overshoots.

Chunked cross-entropy is required at this scale — the 248k-vocab logits tensor is ~14 GB
at 88 k tokens. The segment plan named it as the first escalation lever and never needed
it; here it is the difference between fitting and not.

N_max stays a CLI flag. Every number above is an estimate until Phase 3.

### 2.3 Question-anchored sampling — the cheap partial answer to §1.3

27.8% of questions quote a timestamp (§1.4). For those rows, spend **half the frame
budget densely around the quoted timestamp(s)** and the rest uniformly over the prefix.

On *"There is one Sponge in the frame at 00:09:19. When is it retrieved?"* over a 3-hour
window, the plain N_max=768 grid is past its 64-minute 5 s budget and stretches to ~28 s
anchor spacing everywhere. The hybrid schedule instead gives true 5 s density around
00:09:19 and degrades the rest to ~56 s, where nothing is being asked. It is retrieval
without a retriever: no second model, no extra tokens, no extra latency — just a
different index list.

**The sampler becomes question-dependent, which changes where it runs.** In segment it
was a function of the window alone, so the export could precompute `frames_indices` and
the submission wrapper could reproduce them from `[start, end]`. Here it is a pure
function of `(window, question_text, fps, N)` and must run in **two** places:

- **at export time**, so training and local-eval rows carry a stored `frames_indices` —
  reproducible, and no parsing cost in the dataloader;
- **at inference in the container**, live on the incoming question, because there is no
  export there. The request carries `question`, `start_time`, `end_time`; that is
  everything the function needs.

One implementation called from both, never two. It must be **deterministic** — the same
(question, window) must yield byte-identical indices in training and in the container,
or every anchored row is a silent train/inference mismatch.

Four rules it must obey:
- **Parse the timestamps from the ORIGINAL question, before §2.4 prepends the window
  line.** `Procedure observed from 00:00:00 to 02:47:31.` contains two `hh:mm:ss`
  values; parsing after prepending would make every row look anchored, at the window
  edges, and quietly destroy the policy. Order is load-bearing.
- **Indices stay absolute, monotone, and even in count.** `temporal_patch_size=2` fuses
  pairs; a dense block spliced into a uniform grid must not break monotonicity or the
  rendered markers stop ascending.
- **Exactly N after merging.** Sorting and de-duplicating the union of the uniform grid
  and the dense block yields fewer than N whenever they overlap; pad by subdividing the
  widest remaining gaps, so sequence length stays constant and VRAM predictable.
- **Clamp the quoted timestamp into `[0, t]`** before use, and fall back to plain uniform
  when parsing finds nothing — that is 72.2% of rows.

*Not adopted for the first submission:* end-weighted sampling. `time` answers sit at a
median 0.686 of elapsed time with p95 0.993 (§1.3), so biasing density late is tempting
and free. It is left out because it would help "last visible / retrieved" questions and
hurt "first visible" ones, and nothing in the data says which way the net lands. It is
the first ablation (§4).

### 2.3a Timestamps: text markers **and** the burned-in clock

The segment track chose markers over the overlay, and the deciding row in its table was
*"at submission time we must burn the clock ourselves on raw challenge video with
byte-identical draw params."* **That objection is void** — the organizers ship
`overlayed/<qID>.mp4` with `HH:MM:SS` already burned in, showing *"the source
timeline's second (floored)"* (§1.6). Use both mechanisms.

**Why the overlay matters more here than it did on segment.** The segment plan's own
results concluded the residual `time` error is *not* anchor spacing but the model
interpolating between anchors and converting to `hh:mm:ss`. Markers require associating
a text label with a frame; with 40 markers over 5 minutes that is tractable, with 384
markers over hours it is a materially different task. The burned-in clock removes the
association entirely — the timestamp is *in the frame the model is looking at* — and it
is already in the answer's own format, so no divmod. It is also **immune to any error in
our index arithmetic** (§2.1), which makes it a second, independent line of defence.

**Cost of keeping the markers too: ~4%.** Measured on the real tokenizer — 8 tokens for
`<559.8 seconds>`, 10 for `<17779.6 seconds>`. Markers scale as N/2 while frames scale
as N, so the share is constant at any N_max: 3,840 marker tokens against 84,480 video
tokens at N_max=768. Keep them; they fail differently from the overlay and cross-check
it. Do **not** hand-roll timestamps into the prompt text — `Frame 12 at 00:09:19:` costs
16 tokens, double the processor's, and breaks the interleaved marker→frame pattern
Qwen3-VL was pretrained on.

**Legibility is verified.** The clock is drawn at a *fixed pixel size* (~36 px tall,
~207 px wide) regardless of native resolution, so its relative size varies 2× across the
dataset. Measured glyph height after resizing to 640×360
(`overlay_legibility.ipynb`):

| native | videos | glyph at 640×360 |
|---|---|---|
| 1280×720 | 77 lapchole | **19 px** ← worst case |
| 720×576 | 6 | 22 px |
| 960×540 | all 30 heico | 24 px |
| 720×480 | 6 | 26 px |
| 640×360 | 5 | 38 px |

19 px of digit height at the worst end is comfortably readable, so the overlay arm is
safe. Two secondary results: the challenge's height-normalise-then-resize path lands on
**identical** geometry to resizing directly, so the train/inference mismatch worried
about in §1.5 does not materialise; and if the organizers burn the clock *after*
normalising, the 1280×720 case becomes 23 px rather than 19 — a 20% difference, not a
problem. The notebook also sweeps frame sizes below 640×360, for if VRAM ever forces
one.

### 2.4 Put the window end in the prompt

Prepend `Procedure observed from 00:00:00 to 02:47:31.` to the question. Ten tokens.

100% of windows start at zero and only the end varies, and 46.5% of repeated
(video, question) pairs change their answer with it (§1.2). Every "so far" question
needs that denominator, and the frame markers give it only implicitly — as the largest
value in a 128-marker list the model has to notice.

This was an un-run ablation in the segment plan, where the window was incidental. Here
the prefix *is* the variable, so it goes in by default.

### 2.5 Prompt: extend `../orena_sft/prompts.py` again

`build_system_prompt(track=…)` already has `frame` and `segment` arms. Add `procedure`
rather than reusing `segment`:

- the `<SECONDS seconds>` → `hh:mm:ss` instruction stays verbatim — it is the same
  mechanism and it demonstrably trains (segment `time` .21 → .49)
- wording moves from "a short clip" to "a procedure observed from its start up to a
  given time", and states that **nothing after that time is observable** — half the
  questions are "so far" questions and the model must not extrapolate
- counting guidance for `number`, which triples in share and whose answers reach 76
- keep `percentage`, multi-timestamp and FO-vocabulary rules unchanged

*Verify:* `build_system_prompt(track="frame")` and `track="segment"` outputs stay
byte-identical — they guard two shipped models.

### 2.6 Folder layout

```
procedure_track/
  plan.md                            ← this file
  build_procedure_sft_dataset.py     export → train/eval/test.jsonl
  sft_train_qwen_procedure_ddp.py    + .slurm
  evaluate_procedure.py              + .slurm   (fork of evaluate_base_segment.py)
  preflight.py
  tests/
  sft_export/
  checkpoints/
  logs/
```

Imported, not copied, from `../segment_track/`: `clip_sampling.py` and `collate.py`
(§2.1). From `../orena_sft/`: `prompts.py` and `make_eval_video_split()`. If this
third track makes the `sys.path.insert` pattern painful, that is the moment to promote
the shared modules to a real package — not before.

### 2.7 I/O — the page cache will not save us this time

The segment track got away with reading JPEGs off `/projects` because the ~128 GB of
reads had heavy frame reuse between overlapping windows and fit behind a 256 GB `--mem`
request. Procedure reads **205 GB per epoch** (10,000 × 256 × ~80 KB) drawn from a
**1.24 TB** pool (§1.5), and prefixes share their early frames but diverge over hours,
so residency is not achievable.

What still holds: the path is **latency-bound, not bandwidth-bound** (~71 ms/frame cold,
9 ms warm), so parallelism is the lever. One optimizer step at N_max=768 is at most 16
micro-batches × 768 = 12,288 frames per GPU — ~15 min single-threaded, ~27 s across 32
CPUs, against an estimated 270 s step. It should still hide, but with far less margin
than segment had, and the true figure is lower because most rows use fewer than 768
frames. Log dataloader wait against GPU step time in the smoke run and treat a per-clip
repack to node-local scratch as a live option, not a theoretical one.

### 2.8 Decoding in the container — largely resolved

The original worry was seeking 256 frames across a raw 5-hour file, and it was named the
most likely way this plan's latency budget turns out wrong. The released template
retires most of it (§1.6): the clip is **pre-trimmed**, **5 fps**, **≤576 px tall**, and
carries **a keyframe every 5 s**.

At 5 fps that is a keyframe every 25 frames, so a seek costs at most 25 decoded frames
and ~13 on average:

| | frames decoded |
|---|---|
| 768 scattered seeks | ≤ 19,200 |
| sequential pass over a 5-hour clip at 5 fps | 90,000 |

**Seek; do not stream.** This inverts the plan's original fallback — a sequential decode
pass, which was the escape hatch, is now the slow path by ~5×. What remains worth
measuring is the absolute number on real files (Phase 0), since it lands in `prep_time`
and the budget is pooled rather than per-question.

---

## 3. Implementation steps

### Phase 0 — the checks that could invalidate §2

1. **✅ DONE — the 5 fps round-trip.** `verify_fps_roundtrip.py` encodes a real training
   window to a 5 fps MP4, decodes it back, runs the real processor, and asserts the
   markers match the training path. **All pass**: 565.0 vs 565.0 exactly. Both failure
   modes confirmed detectable — wrong fps is 5× off, an unshifted origin costs the full
   window offset on segment and is harmless on procedure (§2.1). Re-run it before every
   submission; a wrong marker is invisible to local evaluation.
2. **✅ DONE — overlay legibility.** `overlay_legibility.ipynb`. Glyph height at 640×360
   is **19–38 px** depending on source resolution, worst case on the 77 lapchole videos
   that are natively 1280×720. Readable throughout; the overlay arm is safe (§2.3a).
3. **Decode cost on a real clip.** Time 768 scattered seeks against a sequential pass on
   a 5 fps, ≤576p, 5-hour MP4 with keyframes every 5 s. §2.8 predicts seeking wins by
   ~5×; confirm the absolute number lands well inside the pooled budget.
4. **Confirm frame coverage and fps** for all 130 procedure videos, as the segment
   builder does — assert each video's real fps equals `DATASET_BASE_FPS`, since every
   exported timestamp is `frame_index / fps` and a 29.97-vs-30 drift is silent.
   *Expected:* full coverage (§1.5).

### Phase 1 — export: `build_procedure_sft_dataset.py`

3. Fork the segment builder. Per row: `[0, t]` → N absolute indices via the hybrid
   sampler (§2.3), clipped to frames on disk, N even, repeated indices for windows too
   short to hold N distinct frames.
4. Record the segment fields plus `window_end`, `n_quoted_timestamps`, and which
   sampling policy the row used — the ablation in §4 needs to slice on it.
5. Prepend the window line to the question text (§2.4).
6. **Video-level, procedure-stratified eval split**, `--eval-frac 0.10`, seed 42, via
   `make_eval_video_split()` unchanged. Prefixes of one video overlap almost totally, so
   a row-level split would leak completely — the same rule as segment, harder.
   *Verify:* 10,000 rows total, no video in two splits, every `frames_indices` entry on
   disk, indices monotone and inside `[0, t]`, format distribution matching §1.1.

### Phase 2 — prompt

7. Add the `procedure` arm to `../orena_sft/prompts.py` (§2.5).
   *Verify:* the `frame` and `segment` outputs are byte-identical to today's.

### Phase 3 — training

8. **Smoke run, `--max-steps 20`, N_max=768.** This is where §2.2's estimates become
   numbers: peak VRAM, step time, and **dataloader wait against GPU step time** (§2.7).
   *Escalation order if VRAM does not fit:* chunked cross-entropy → N_max=512 → frame
   size → LoRA rank. Extrapolate the step time against the 2-day wall **before**
   launching.
9. **Honest-split run** on the 6,873-row train split, eval every 50 steps on a
   stratified subset, and sweep checkpoints on a fixed stratified test subsample. This
   picks the step count — it is the only reason the segment submission knew when to stop.
10. **Submission run:** retrain on all 10,000 rows at the same epoch fraction, early
    stopping disabled, length fixed by `--max-steps`. Add the `--include-test` flag to
    the builder rather than concatenating files by hand — the segment track shipped a
    model whose training data came from an unrecorded shell command, and that should not
    happen twice.

### Phase 4 — evaluation and packaging

11. Fork `evaluate_base_segment.py` (sharding, resume, warmup, prefetch) and call
    `Evaluator.run(..., track=Track.PROCEDURE)` so the **30 s** ceiling applies.
    Non-negotiable: same N_max, frame size, sampler policy, frames folder and prompt arm
    as training.
12. **Baseline first.** Score the base model before reading any SFT number. Break
    accuracy out by `answer_format`, by capability group, and — new for this track — by
    **window duration**, since a 26-minute lapchole prefix and a 4-hour heico prefix are
    different problems sharing one score. Also split by `n_quoted_timestamps`, to check
    the §2.3 anchored rows actually beat the unanchored ones.
13. Report heico test (Sigmoid Resection) separately as the OOD number.
14. **Build the submission wrapper against the real template**, shared with the segment
    track: `request.json` → sampler → seek from `overlayed/<qID>.mp4` → resize →
    `VideoMetadata(fps=5.0, …)` → generate → `answer.json`. Validate it on locally
    re-encoded 5 fps clips scored with the Evaluator *before* spending a submission.
    Debug it against the finished segment model, whose correct answers are already known.

### Phase 5 — write-up

15. `method_procedure.md` and a `journals/` entry with the measured decode cost, token
    counts, VRAM, step time and per-bucket accuracy. The segment track never wrote its
    equivalent and its entire result set lives in CSVs; do not repeat that.

---

## 4. Risks and open questions

| risk | severity | status |
|---|---|---|
| `time` (36.1%) is unreachable — required context is 1.5× the model's position budget (§1.3) | **high, structural** | Accepted for the first submission, and partly narrowed: N_max=768 puts **56.6% of rows** at true 5 s density (§2.2), §2.3 recovers the 27.8% of anchored questions, and §2.3a removes the interpolation step for all of them. The long heico windows remain the weak point — report `time` split by window duration |
| Timestamps come out clip-relative or 5× wrong at submission (§2.1) | was **high and silent** | **Resolved.** `verify_fps_roundtrip.py` passes on a real 5 fps encode/decode, and both failure modes are proven detectable. Re-run before every submission |
| Overlay clock illegible after resize (§2.3a) | was medium | **Resolved.** 19–38 px glyph height at 640×360, measured across every native resolution in the dataset |
| Decode blows the budget (§2.8) | was **high, untested** | **Largely resolved** by the released encoding spec — pre-trimmed, 5 fps, keyframes every 5 s make seeking ~5× cheaper than a sequential pass. Phase-0 step 3 confirms the absolute number |
| Frames do not fit the page cache — 1.24 TB (§2.7) | medium | 32 CPUs of dataloader parallelism; per-clip repack to node-local scratch is a live fallback, not theoretical |
| N_max=768 OOMs at 141 GB | medium | chunked cross-entropy first (the ~14 GB logits tensor), then N_max=512 |
| `number` triples in share with answers up to 76, needing event-level coverage | medium | expect this bucket to underperform segment; it is the clearest reason a retrieval stage would pay |
| Prefix overlap inflates the effective epoch — 33 windows per video sharing early frames | medium | video-level split (already); consider deduplicating by (video, question) and treating windows as augmentation. Unresolved in segment too |
| **Only 10–11 submissions and ~3 weeks**, on a tier that queues 12 h–3 days (§1.6) | **high, schedule** | Build the wrapper once and share it with segment; submit the finished segment model first for free information about where the baselines sit; make each procedure shot the configuration actually believed in |
| 10,000 rows total, complex reasoning is 219 of them | low | do not tune on the small buckets; expect noise |

**Settled:**
- **Reuse over rewrite** — one sampler, one collator, one trainer across segment and
  procedure, all working in the 5 fps index space (§2.1).
- **Markers *and* the burned-in clock** (§2.3a) — the segment track's overlay verdict is
  reversed, because the organizers now ship the clock and because the overlay attacks
  the error mode segment's own results identified as binding.
- **Qwen3.5-9B at N_max=768**, capped-5s + anchored. Not a VRAM constraint — 96 GB fits
  the 27B — but at equal training cost the 9B buys 3× the temporal resolution on the one
  axis this track is hard along (§2.2).

**Still open, in priority order:**

1. **End-weighted sampling** (§2.3). Free, and the answer-position skew is real
   (median 0.686, p95 0.993) — but it trades "last visible" against "first visible" and
   the data does not say which wins. Now the cheapest remaining experiment.
2. **Overlay-only vs markers-only vs both.** §2.3a argues for both on the grounds that
   they fail differently, but that is reasoning, not measurement. One export flag each.
3. **A retrieval stage — the real answer to §1.3, and a second-submission problem.**
   Two-stage locate-then-read: a cheap pass over a dense frame grid to propose candidate
   windows, then the VLM on a narrow window at segment-like resolution. This is the only
   design that can actually reach a 5 s tolerance over 5 hours. Do not start it before
   the first submission gives a baseline to measure it against.
4. **Does the segment adapter transfer?** The two tracks share a schema, a prompt family
   and a marker mechanism. Initialising from the shipped segment LoRA, or training one
   multi-track adapter, is cheap to try and would answer whether the compliance gain has
   to be re-learned per track.
