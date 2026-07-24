# ORena SAVE FOCUS Challenge — FRAME Track: Task Description & Approach Plan

*Compiled from the challenge website (frame.orena-focus-challenge.org), the `orena-focus` package README, and the HeiCo-FOCUS dataset. Last checked 2026-07-14.*

---

## 1. Challenge context

ORena SAVE FOCUS is a MICCAI 2026 challenge on **Foreign Object Contextual Understanding for Surgery**: whether vision-language models can detect, count, and reason about foreign objects (sponges, needles, clips, drains, specimen bags, etc.) in laparoscopic/endoscopic video, in support of intraoperative safety and retained-foreign-object prevention. It has three independent tracks that share one capability taxonomy and evaluation protocol but differ in visual context length:

| Track | Visual input | What it tests |
|---|---|---|
| **FRAME** (this track) | Single extracted video frame | Perception from one image — no temporal modeling |
| SEGMENT | Clip up to 5 min | Local temporal reasoning, short-term tracking |
| PROCEDURE | Up to a full procedure | Long-horizon memory, persistent tracking, retrieval-status reasoning |

FRAME is explicitly positioned as the "simplest entry point" — you can enter any subset of tracks, and FRAME is a reasonable one to focus on first.

---

## 2. Data

### 2.1 Sources and scale

| | Batch 1: HeiCo-FOCUS | Batch 2: LapChole-FOCUS | Combined |
|---|---|---|---|
| Procedure type | Colorectal surgery (proctocolectomy, rectal resection, sigma resection) | Laparoscopic cholecystectomy | — |
| Videos | 30 | 170 | 200 |
| Total VQA pairs (all tracks) | 15,000 | 35,000 | 50,000 |
| FRAME-specific VQA pairs | 6,000 (4,000 train / 2,000 held-out within batch) | 14,000 | ~20,000 |
| SEGMENT-specific | 6,000 | 14,000 | 20,000 |
| PROCEDURE-specific | 3,000 | 7,000 | 10,000 |

Both batches are on Hugging Face (`orena-dkfz/heico-focus-vqa`, `orena-dkfz/lapchole-focus-vqa`) and are fetched automatically by the `orena-focus` Python package. Note: the Batch-1 "test" split above (10 sigma-resection videos) is an *internal* held-out split within the released training data — it is not the official challenge test set, which is never released to participants.

**HeiCo-FOCUS internal split** (Batch 1, as originally scoped): training on 8 proctocolectomy + 8 rectal-resection videos (internal train), 2+2 held out as internal validation, and 10 sigma-resection videos held out as an internal "unseen procedure type" test. The equivalent internal split for the 170 LapChole videos is not published on the pages checked — treat as unknown until confirmed via the dataset card or package.

### 2.2 Official evaluation data (never released to participants)

| Phase | Videos | Notes |
|---|---|---|
| Pre-evaluation | 20, mixed procedures, unseen | Public leaderboard, ID + OOD questions |
| Final test | 200, mixed procedures, unseen | Only reachable after beating both baselines in pre-eval |

### 2.3 Capability taxonomy

Five capability groups exist across the whole challenge, but **FRAME uses a reduced taxonomy limited to what a single image can support**: object recognition and aggregation. The other three groups (temporal grounding, event/procedural understanding, most of complex reasoning) inherently require multiple frames or time and are SEGMENT/PROCEDURE territory.

| Group | Leaf capabilities | In FRAME? |
|---|---|---|
| 1. Object Recognition | Identification, Instance Matching, Attributes, Spatial (camera), Spatial (situs) | Yes |
| 2. Temporal Grounding | Temporal Localization, Duration Estimation | No |
| 3. Aggregation | Object Aggregation, Event Aggregation | Object Aggregation only |
| 4. Event & Procedural Understanding | FO Interaction Recognition, FO Usage Purpose, Temporal Ordering | No |
| 5. Complex Reasoning | Functional Reasoning, Causal & Consequence Reasoning, Multi-step Reasoning | Largely no |

Each test case also carries a **robustness label**: in-distribution (ID) or out-of-distribution (OOD) — OOD meaning an unseen procedure type or an unseen question phrasing. Final scoring buckets are capability × robustness, so FRAME has fewer buckets than SEGMENT/PROCEDURE (roughly 2 capability groups × 2 robustness levels).

### 2.4 Answer formats

Every question declares its answer format up front; the format determines both what a valid response looks like and how it's scored. Implemented in `focus.data.formats`:

| Format | Accepted response | Scoring |
|---|---|---|
| `binary` | `yes`/`no` (case-insensitive) | Exact match on parsed bool |
| `number` | Non-negative integer | Exact match |
| `percentage` | Number, optional `%` | Tolerance-aware match (`threshold_pp`) |
| `fo_class` | Registered foreign-object class name or `none` | Case-insensitive exact match |
| `time` | `hh:mm:ss` | Tolerance-aware match (`threshold_seconds`) — likely SEGMENT/PROCEDURE only, since FRAME has no duration/localization questions |
| `multiple_choice` | One of a predefined option set | LLM-as-a-judge |
| `open_ended` | Free text, ≤300 chars | LLM-as-a-judge |
| `matching` | Regex-validated text | LLM-as-a-judge |

A critical, easy-to-miss point: **closed-form formats are scored by exact string/value match after format-specific parsing** — a semantically correct but malformed answer (e.g., "2 clips" instead of "2" for a `number` question, or "No" not matching due to unexpected punctuation) is scored as **incorrect**. Output formatting discipline matters as much as visual correctness.

### 2.5 Input structure

Per the track overview page, each FRAME test case gives the model: the RGB frame, metadata (procedure type, timestamp of the frame within the source video), and the question text. Exact serialization (JSON schema, prompt template) will follow the official submission template repository, not yet released.

---

## 3. Pipeline (as supported by the `orena-focus` package)

```
pip install orena-focus
```

```python
from focus import download, FocusDataset, DatasetSplit, Track
from focus.preprocessing import VideoTimestampOverlayPreprocessor, FrameExtractorPreprocessor

download("heico")
VideoTimestampOverlayPreprocessor().process(dataset="heico")   # burns/exposes timestamp metadata
FrameExtractorPreprocessor(stride=1).process(dataset="heico")  # extracts frames from source video

ds = FocusDataset("heico", DatasetSplit.TEST, Track.FRAME)
request, reference = ds[0]
```

`request` carries the question (+ metadata); `reference` carries the ground-truth answer and its `format`. QA annotations are pulled automatically from Hugging Face; you control video → frame extraction (stride) yourself.

For local development, evaluation is also provided:

```python
from focus import Evaluator, Response
responses = [Response(qID=req.qID, content=my_model(req)) for req, _ in ds]
results_df, summary_df = Evaluator().run(requests=ds.requests, references=ds.references, responses=responses)
```

This lets you reproduce per-case and per-bucket scoring **on the released training/internal-val data** before spending one of your 10 pre-evaluation submission slots. A sample judge implementation for the open-ended formats is in `focus.evaluation.judges`, and a Qwen3-VL inference example is provided in `examples/inference.py` in the repo — worth reading before building a custom harness.

**End-to-end pipeline for FRAME:**

1. Download HeiCo + LapChole batches, extract frames at the relevant timestamps.
2. Build (image, metadata, question) → answer training examples, formatted for whichever VLM's chat template you target, tagging each example with its declared answer format so the model learns the right output shape per format (e.g., bare "yes"/"no", bare integer, a class name from the registered FO vocabulary).
3. Train (see Section 5).
4. Score locally on internal validation split using `Evaluator` to estimate bucket-level accuracy before submitting.
5. Package as a Docker image satisfying the inference constraints (Section 4), test the 5s/question budget locally.
6. Submit to the pre-evaluation leaderboard (10 submissions max); iterate.
7. If mean accuracy beats **both** baselines, advance to final test.

---

## 4. Constraints

| Constraint | Detail |
|---|---|
| Inference time budget | 5 seconds per question |
| Inference hardware | Single GPU, 48GB VRAM |
| Container | Docker, **no internet access** at inference — everything (weights, judge if you bundle one, etc.) must be self-contained |
| Submission cap | Up to 10 submissions during pre-evaluation (may be adjusted) |
| Missing/timeout | Any case without a response, including timeouts, scored as incorrect |
| Allowed training resources (FRAME/SEGMENT specifically) | Challenge data, **public** datasets, **public** pre-trained models only. All resources must be disclosed and must have been publicly available by the start of the pre-evaluation phase. (PROCEDURE track has looser rules — private data and closed models allowed there, not here.) |
| Qualification for final test | Must beat **both** provided baselines (a frontier closed-source VLM run zero-shot, and an organizer-fine-tuned open-source VLM) on mean accuracy across buckets during pre-evaluation |
| Anti-gaming | Adversarial prompting/jailbreaking targeted at the LLM-judge results in disqualification |
| Publication | Teams beating baselines may be invited as co-authors (up to 3 per team) |

The "public pre-trained models only" rule is worth flagging now: it clears Qwen2.5/3-VL, Gemma, DINOv3, CLIP, etc. (all public open-weight), but raises a genuine open question about whether using a *closed* frontier model purely as a reward signal during GRPO training (not for distillation of labeled data) would count as an undisclosed/non-public "training resource" under this rule. Worth asking on the challenge forum before building around it.

---

## 5. Metrics and ranking

- **Primary metric: Accuracy.** Closed-form questions (`binary`, `number`, `percentage`, `fo_class`, `time`) are scored by exact match after format-specific parsing, with tolerance windows for `time`/`percentage`. Open-form questions (`multiple_choice`, `open_ended`, `matching`) are scored by **up to 3 undisclosed LLM judges, majority vote** (short-circuits once a majority is reached).
- **Aggregation:** mean accuracy computed per (capability × ID/OOD) bucket, so a track with many "easy" questions in one bucket can't dominate the score.
- **Ranking (final phase only):** per-bucket ranks with significance-adjusted rank collapsing → combined via the **Copeland method** (pairwise domination counts) → bootstrap win-rate tie-break for top-3 ties. Pre-evaluation qualification, however, is decided by simple mean accuracy across buckets versus both baselines — the Copeland ranking only matters once you're already in the final phase.

---

## 6. Your three proposed approaches: implementation, pros, risks

All three assume a public, open-weight starting point per the resource rule, and all need to hit the 5s/48GB inference budget and produce format-compliant short-text answers, not just semantically-correct ones.

### A. SFT on an existing VLM (Qwen2.5/3-VL, Gemma, etc.)

**Implementation.** Take a public instruction-tuned VLM in the 3–8B range (comfortably fits 48GB with headroom for KV cache and generation at ≤5s/question; larger, e.g. 30B-class, generally requires quantization to fit the budget, as the organizers' own Gemma-31B-GGUF baseline does). Fine-tune with LoRA/QLoRA rather than full fine-tuning given the training set size (~20k FRAME-specific QA pairs, fewer once split into train/val). Format each example as image + metadata (procedure type, timestamp) + question → answer, and, importantly, **train the model to emit exactly the accepted string for the declared format** (bare "yes"/"no", bare integer, exact FO class name from the registered vocabulary, etc.) rather than a hedged or explanatory sentence, since the closed-form scorer does literal parsing. Use the internal HeiCo validation split (2+2 videos) and a held-out slice of LapChole for model selection; reserve the sigma-resection videos as your own OOD-generalization proxy, since they represent an unseen procedure type, similar in spirit to the official OOD bucket.

**Pros.** Fastest to implement with mature, well-documented tooling (HF `transformers`/`trl`, LoRA libraries). Leverages a projector and vision-language alignment that's already been trained on web-scale image-text pairs — you're only adapting to a new visual domain and output format, not learning multimodal grounding from scratch. Directly optimizes the thing the primary metric rewards (token-level correctness on the target string). Straightforward to package into a Docker image and control latency, since it's one forward pass per question.

**Risks.** ~20k FRAME QA pairs (even less once split train/val, and further split by capability) is a fairly small SFT set for a full LLM backbone — risk of overfitting to HeiCo/LapChole phrasing and imagery, hurting OOD accuracy on the truly unseen procedures/phrasings in the official test buckets, which are explicitly weighted equally to ID. Catastrophic forgetting of general visual/language competence is a real risk with aggressive full fine-tuning (mitigated by LoRA, but LoRA underperforms full FT in some domain-shift settings). The submission template's exact expected format isn't published yet, so early SFT formatting choices may need rework once it lands. Also, `fo_class` questions require agreement with a *registered* FO vocabulary — if your training data doesn't cover the full class list, the model may fail on classes seen rarely or not at all in training.

### B. GRPO, optionally SFT-init + GRPO

**Implementation.** Start from an SFT checkpoint (recommended over GRPO-from-scratch — provides a stable initial policy and lets GRPO focus on refinement rather than learning task format from a cold start). Define reward functions per answer format: for `binary`/`number`/`fo_class` you have clean, verifiable rewards (exact match against the label, no judge needed); for `percentage`/`time` a tolerance-shaped reward mirroring the official tolerance window; for `multiple_choice`/`open_ended`/`matching` you'd need either a heuristic proxy reward (e.g., keyword/embedding similarity to the reference) or an actual judge model in the loop, which is far more expensive per rollout and only a proxy for the real, undisclosed judge ensemble. Use group sampling (multiple completions per prompt), reward normalization within the group, and a KL penalty against the SFT reference policy to avoid drift. `trl`'s GRPOTrainer is a reasonable starting point rather than a from-scratch RL loop.

**Pros.** Directly targets what the leaderboard is graded on — accuracy-shaped behavior — for the categories with clean, verifiable rewards, which happen to be most of FRAME's closed-form question types. Zero added inference cost or latency, since the deployed model is the same size/architecture as after SFT — no separate reward model runs at test time. Can improve robustness on harder within-frame reasoning (e.g., ambiguous instance-matching or attribute cases) beyond what supervised cross-entropy alone teaches, since it optimizes end-behavior rather than next-token likelihood on a single reference phrasing.

**Risks.** Meaningfully more engineering than SFT: multiple generations per prompt, reward computation, RL training stability (collapse/degenerate outputs are common failure modes, especially on a training set this size). Reward hacking is the central risk for anything relying on a proxy judge for open-ended/multiple-choice/matching reward — the model can learn to game your proxy without actually improving against the real, undisclosed judge ensemble used for the leaderboard. Using a closed frontier model as that proxy judge also runs into the "public pre-trained models only" resource-disclosure question flagged in Section 4 — worth clarifying with organizers before investing here. Small prompt set (a few thousand FRAME training questions) limits rollout diversity and can make GRPO's group-relative advantage estimation noisy. Net effect: GRPO is a high-value refinement step on top of SFT for the verifiable-reward question types, but a shakier bet for the LLM-judged ones without more infrastructure than the closed-form gains might justify early on.

### C. Pretrained visual encoder (DINOv3, CLIP) + trained projector/LLM, possibly unfreezing the encoder

Two meaningfully different implementations hide under this idea — worth separating them:

**C1 — Adapt the vision tower inside an already-aligned VLM.** Most VLM LoRA recipes freeze the vision encoder and only adapt the projector + LLM. Since laparoscopic/endoscopic imagery is a real domain shift from the web-scale natural images DINOv3/CLIP/SigLIP were pretrained on, you additionally apply LoRA (or unfreeze the last few blocks) of the vision tower inside your chosen VLM (from option A), with a low learning rate, layered on top of the existing SFT/GRPO training.
*Pros:* incremental on top of Option A rather than a separate build; directly targets the domain gap most likely to be limiting perception (surgical scenes, smoke, specular reflections, unusual object shapes for foreign objects, none of which resemble ImageNet/LAION distributions); modest additional engineering.
*Risks:* with only ~20k FRAME images (fewer unique frames than questions, since multiple questions share frames), full or deep unfreezing of the encoder risks overfitting/collapsing general visual features that the projector and LLM were jointly trained against, which can silently degrade performance on OOD test cases even as training loss improves; needs careful, staged unfreezing (projector only → +LoRA on last encoder blocks) and close validation-set monitoring to catch this early.

**C2 — Build a custom architecture from a separate encoder + LLM (LLaVA-style, from scratch).** Pick a strong frozen or lightly-adapted vision backbone (DINOv3 for dense/spatial tasks such as instance matching and spatial localization, or CLIP/SigLIP for more semantic attribute tasks — potentially both, fused via a multi-encoder projector), attach a trainable projector (MLP), and a separate LLM decoder (open, public, sized to fit the inference budget), training the whole stack on the FRAME data with staged unfreezing.
*Pros:* full control over architecture and combining encoders' respective strengths (dense spatial features vs. semantic ones); smaller total parameter count is achievable, giving comfortable latency margin; not locked into a general-purpose VLM's existing multimodal alignment quality/blind spots.
*Risks:* this is a much bigger engineering lift than A/B/C1 and, more importantly, a vision-language *alignment* step (getting the projector to map encoder features into the LLM's embedding space at all) typically needs large-scale image-caption/instruction data before task fine-tuning even starts — this dataset is VQA-only, not a captioning corpus, so building alignment from scratch on ~20k narrow QA pairs is likely to underperform a VLM that arrives already aligned on billions of image-text pairs (which is exactly what option A starts from). Given the challenge timeline and the fact that "complex reasoning"-adjacent question phrasing still benefits from a well-instruction-tuned LLM backbone, C2 is best treated as a stretch/parallel research direction rather than the primary path.

---

## 7. Suggested sequencing

Given the constraints above (small-ish FRAME-specific training set, hard latency/VRAM ceiling, no-internet Docker, OOD-weighted scoring, unpublished submission template), a reasonable path is:

1. **Baseline fast:** LoRA SFT (Option A) on a mid-size public VLM (e.g., Qwen2.5/3-VL 7B), correctly formatted per answer type, validated on the internal HeiCo val split + sigma-resection OOD proxy. Get this through the local `Evaluator` and a first pre-evaluation submission early, since you only get 10 submissions total and the submission template itself may reveal formatting requirements you need to react to.
2. **Layer in C1** (staged vision-tower adaptation) if validation error analysis shows perception/localization errors dominate over language/format errors.
3. **Layer in GRPO (Option B)** on top of the SFT+C1 checkpoint, focused first on the verifiable-reward question types (`binary`, `number`, `fo_class`, and tolerance-shaped `percentage`), leaving LLM-judged categories for later once/if the resource-rule question around proxy judges is resolved.
4. **Treat C2** as a parallel exploratory track only if time/compute allow, not the critical path to a first competitive submission.

---

## 8. Open questions to resolve before implementation

- Exact FRAME train/val split for the 170-video LapChole batch (not found on the pages checked — confirm via dataset card or package internals).
- Exact input/output serialization for submissions — the official submission template repository is not yet released.
- Whether a closed frontier model used purely as a GRPO reward signal (not for data generation/distillation) is compatible with the "public pre-trained models only" resource rule for FRAME — worth asking on the challenge forum.
- Full registered foreign-object class vocabulary for `fo_class` questions, and its coverage in the training data vs. what might appear OOD.
- Track-specific design document (FRAME PDF linked from the rules page) may contain additional detail on question generation methodology and annotation process not covered on the HTML pages — worth a closer read before finalizing the data pipeline.
