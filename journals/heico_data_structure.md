# FOCUS Data Structure — General Overview & Heico Deep Dive

This journal describes the two layers of data used in this project, then drills
into the specifics of the **Heico** dataset: video/frame counts, capability
distributions, answer-format distributions, and example question/answer pairs
for every capability. It mirrors what `data_exploration.ipynb` computes, but
in written form.

---

## 1. General data structure

There are two independent data sources per procedure family (`heico`, `lapchole`):

### 1.1 Raw visual data (local, on `/projects/datasets_ML/orena/<dataset>/`)

Configured via `focus.FocusConfig(root_dir=...)` / `set_config(...)`. Layout:

| Folder / file | Contents |
|---|---|
| `videos/` | Raw source surgical videos (`.avi`), one per procedure, named `"<idx> - <Dataset> - <ProcedureType> - <n>"`. |
| `overlayed/` | Same videos re-encoded as `.mp4` with a burned-in timestamp overlay (`FocusConfig.OVERLAY_FOLDER`). |
| `frames/` | Extracted individual frames on disk, one subfolder per video (`FocusConfig.FRAMES_FOLDER`). |
| `frames_overlay/` | Extracted frames from the overlaid videos. |
| `train.lance/`, `val.lance/`, `test.lance/` | Columnar (Lance-format) tables of **individual JPEG frames**, one row per frame, split at the *video* (episode) level so no video's frames leak across splits. Schema: `episode_idx (int32), step_idx (int32), frame (binary/JPEG), h (int16), w (int16)`. |
| `train.lance.episodes.json`, `val.lance.episodes.json`, `test.lance.episodes.json` | Index mapping each `episode_idx` back to its source video (`stem`), frame count (`T`), and starting row offset (`start_row`) in the corresponding `.lance` table. |
| `internal_val_videos.json` | Fixed, seeded (seed=42) list of video IDs held out for internal validation. |
| `__manifest/` | Lance's internal versioning/transaction bookkeeping — not meant to be read directly. |

This layer has **no question/answer content** — it is the raw pixel data,
intended for pretraining / self-supervised use or as the visual backing store
for the QA benchmark below.

### 1.2 QA benchmark data (remote, Hugging Face Hub)

Loaded via `focus.FocusDataset(dataset, split, track)`, which always calls
`datasets.load_dataset("orena-dkfz/<dataset>-focus-vqa", config_name, split=...)`
— i.e. it always fetches from the Hub (or its local cache under
`~/.cache/huggingface/`), independent of `root_dir`.

- **Tracks** (`config_name` on the Hub): `frame` (single still frame as
  visual input) and `segment` (short video clip, ≤ 5 min, as visual input).
  A third track, `procedure`, is defined in the taxonomy but not present in
  either dataset's cache.
- **Splits**: `train`, `test` (and `all` = `train+test` concatenation).
- Each row becomes a `(Request, Reference)` pair:
  - **Request**: `qID`, `videoID`, `procedure_type`, `start_time`,
    `end_time`, `duration`, `question`.
  - **Reference**: `answer`, `_format` (answer type), `primary` capability,
    `secondaries` (0+ capabilities), `ood` flag, `clinical` flag.

### 1.3 Capability taxonomy

15 leaf capabilities under 5 top-level groups (leaves are what's actually
assigned to questions; groups are only used for aggregated reporting):

| Group | Leaf capabilities |
|---|---|
| Object Recognition | Object Identification, Instance Matching, Object Attributes, Spatial Localization (Camera), Spatial Localization (Situs) |
| Temporal Grounding | Temporal Localization, Duration Estimation |
| Aggregation | Object Aggregation, Event Aggregation |
| Event Understanding | FO Interaction Recognition, FO Usage Purpose, Temporal Ordering |
| Complex Reasoning | Functional Reasoning, Causal/Consequence Reasoning, Multi-Step Reasoning |

Each question has exactly **one primary capability** and **zero or more
secondary capabilities** (a question can touch multiple skills at once).

### 1.4 Answer formats

Eight formats are defined in the taxonomy: `binary`, `number`, `percentage`,
`fo_class`, `open_ended`, `matching`, `multiple_choice`, `time`. Not every
format necessarily appears in every dataset (see §2.5 — Heico uses 7 of the 8).

---

## 2. Heico dataset — specifics

Analysis below loaded the Heico QA benchmark fully offline from the local HF
cache (`HF_HUB_OFFLINE=1`), both tracks, both splits — **24,000 QA pairs total**.

### 2.1 Videos and frames

**Raw frame data (`.lance`, from `root_dir`)** — split by episode (video), no
QA content:

| Lance split | Videos (episodes) | Frames |
|---|---|---|
| `train.lance` | 16 | 4,995,509 |
| `val.lance` | 4 | 1,249,642 |
| `test.lance` | 10 | 2,436,298 |
| **Total** | **30** | **8,681,449** |

**QA benchmark** — unique videos referenced per split/track (note: the QA
benchmark's train/test split is a *different* partition than the raw-frame
`.lance` train/val/test split above — it's drawn from the same 30-video pool):

| Split | Track | Unique videos | Questions |
|---|---|---|---|
| train | frame | 20 | 8,000 |
| train | segment | 20 | 8,000 |
| test | frame | 10 | 4,000 |
| test | segment | 10 | 4,000 |

30 unique videos overall across the QA benchmark — matching the 30 raw-video
episode count above (16+4+10). All videos belong to one of three procedure
types, evenly split at 8,000 questions each:

- Proctocolectomy
- Rectal Resection
- Sigmoid Resection

### 2.2 Track duration characteristics

- **Frame track**: single still frame → `duration = 0` for every sample
  (12,000 samples).
- **Segment track**: short clips, mean duration ≈ 122 s, median 119 s, range
  1–299 s (i.e. up to ~5 minutes), std ≈ 89 s (12,000 samples).

### 2.3 Primary capability distribution (overall, n=24,000)

| Capability | Count | Track(s) it appears in |
|---|---|---|
| Object Identification | 7,438 | frame (5,266) + segment (2,172) |
| Object Aggregation | 6,075 | frame (4,879) + segment (1,196) |
| Temporal Localization | 4,763 | segment only |
| Spatial Localization (Camera) | 3,150 | frame (1,441) + segment (1,709) |
| Duration Estimation | 452 | segment only |
| Event Aggregation | 452 | segment only |
| Instance Matching | 388 | segment only |
| Spatial Localization (Situs) | 343 | frame (315) + segment (28) |
| FO Interaction Recognition | 192 | segment only |
| Object Attributes | 184 | frame (99) + segment (85) |
| Multi-Step Reasoning | 176 | segment only |
| Temporal Ordering | 129 | segment only |
| Functional Reasoning | 91 | segment only |
| Causal/Consequence Reasoning | 90 | segment only |
| FO Usage Purpose | 77 | segment only |

**Key pattern**: the **frame track only ever uses 5 primary capabilities**
(Object Identification, Object Aggregation, Object Attributes, Spatial
Localization Camera, Spatial Localization Situs) — anything requiring
temporal reasoning, aggregation-over-time, interaction/usage reasoning, or
multi-step reasoning is exclusive to the **segment track**, which makes sense
since those questions need to see change over time.

### 2.4 Secondary capability distribution (exploded across all rows with ≥1 secondary)

22,411 secondary-capability tags across 24,000 questions (1,589 questions
have zero secondary capabilities; a question can carry more than one, so this
sums to more than 24,000):

| Secondary capability | Count |
|---|---|
| Object Identification | 16,434 |
| Object Aggregation | 6,260 |
| FO Interaction Recognition | 2,680 |
| Instance Matching | 2,301 |
| Temporal Ordering | 1,980 |
| Temporal Localization | 1,648 |
| Spatial Localization (Camera) | 1,092 |
| Event Aggregation | 649 |
| Duration Estimation | 310 |
| Object Attributes | 190 |
| FO Usage Purpose | 58 |
| Spatial Localization (Situs) | 35 |
| Functional Reasoning | 18 |
| Causal/Consequence Reasoning | 4 |

**Object Identification dominates as a secondary skill** — almost every
question, regardless of its primary capability, also implicitly requires
identifying the foreign object(s) in view. This is the most "load-bearing"
capability in the taxonomy for this dataset.

### 2.5 Answer format distribution

| Format | Count (overall) | Frame | Segment |
|---|---|---|---|
| `fo_class` | 8,433 | 5,737 | 2,696 |
| `time` | 5,009 | 0 | 5,009 |
| `number` | 4,870 | 3,556 | 1,314 |
| `multiple_choice` | 2,166 | 520 | 1,646 |
| `binary` | 2,086 | 1,316 | 770 |
| `open_ended` | 1,376 | 871 | 505 |
| `percentage` | 60 | 0 | 60 |
| `matching` | **0 — not used in Heico** | — | — |

`time` answers only occur in the segment track (they require a clip to
localize an event in), and `fo_class` (naming a foreign-object class) is the
single most common answer type overall.

### 2.6 OOD / clinical flags

Both `ood` and `clinical` flags are `False` for **every** Heico question in
this cache (mean = 0.0 for both, in both tracks). These flags exist in the
taxonomy but are not exercised in the current Heico snapshot — worth
double-checking against `lapchole` once that dataset is reachable, in case
OOD/clinical subsets are only populated there.

### 2.7 Example question/answer pair per primary (objective) capability

*("FO" = foreign object, e.g. sponge, clip, needle, silicone loop, drain.)*

| Capability | Track | Format | Example Q | Example A |
|---|---|---|---|---|
| Causal/Consequence Reasoning | segment | open_ended | "Is the surgical foreign object visible here typically left intra-abdominally at the end of the procedure? Please answer with 'yes' or 'no'." | "no." |
| Duration Estimation | segment | time | "Adding together all potentially non-consecutive time intervals, for how long is a Sponge visible and the only foreign object class on screen during the video? Please provide an answer in the format hh:mm:ss." | "00:00:01" |
| Event Aggregation | segment | number | "How many separate times does a Sponge completely leave the field of view for 3 frames or more and then return within this video?" | "0" |
| FO Interaction Recognition | segment | binary | "Is the Silicone loop visible at 04:44:15 being inserted in the abdomen at that moment?" | "no" |
| FO Usage Purpose | segment | open_ended | "Does the surgical foreign object visible here usually remain intra-abdominally at the end of the procedure?" | "No." |
| Functional Reasoning | segment | open_ended | "What is the purpose of the clips in this segment? Please provide a function." | "Hold tissue together." |
| Instance Matching | segment | binary | "Does a Sponge leave the field of view for at least 3 seconds and re-enter later?" | "no" |
| Multi-Step Reasoning | segment | fo_class | "With which other foreign object classes does the foreign object class, which is last seen in the video, co-occur throughout the video?" | "Clip" |
| Object Aggregation | frame | number | "How many different foreign object instances appear in this frame?" | "1" |
| Object Attributes | frame | open_ended | "Is the silicone loop visible in this frame currently grasped by an instrument?" | "No." |
| Object Identification | frame | fo_class | "Which combination of foreign object classes is visible in this frame?" | "Clip, Silicone loop" |
| Spatial Localization (Camera) | frame | open_ended | "At timepoint 00:09:39 please provide all relative central positions of foreign objects present in the frame... (quadrant format)" | "1. Sponge: top/right" |
| Spatial Localization (Situs) | frame | open_ended | "Where is the sponge located before becoming not visible for more than one minute?" | "Lower left abdominal quadrant." |
| Temporal Localization | segment | time | "There is one Sponge in the frame at 00:09:19. When is it retrieved from the surgical site?" | "00:10:12" |
| Temporal Ordering | segment | open_ended | "In what chronological order do the different foreign object classes first appear in this video, from first to last?" | "Silicone loop, Clip" |

### 2.8 Example question/answer pair per secondary (sub-)capability

Same 15 capabilities, but this time picking a question where they appear as a
**secondary** skill (primary capability shown for context):

| Secondary capability | Primary (that question's) | Track | Format | Example Q | Example A |
|---|---|---|---|---|---|
| Causal/Consequence Reasoning | Functional Reasoning | segment | open_ended | "Is the surgical foreign object visible here usually left intra-abdominally at the end of the procedure?" | "no." |
| Duration Estimation | Spatial Localization (Situs) | frame | open_ended | "In which abdominal quadrant is the sponge located before being not visible for >1 min?" | "Lower right abdominal quadrant." |
| Event Aggregation | Spatial Localization (Situs) | frame | open_ended | "In which abdominal quadrant is the sponge located before being not visible for >1 minute?" | "(Small) pelvis." |
| FO Interaction Recognition | Object Identification | frame | open_ended | "Is the external drain in this frame currently grasped by an instrument?" | "Yes." |
| FO Usage Purpose | Object Identification | frame | open_ended | "Are there any foreign objects in this frame which should not be removed before the end of the surgery?" | "Yes, an external drain." |
| Functional Reasoning | Object Identification | frame | open_ended | "Are there any foreign objects in this frame that should not be removed before the end of surgery?" | "Yes, an external drain." |
| Instance Matching | Spatial Localization (Situs) | frame | open_ended | "Which structure is the sponge in contact with before disappearing for >1 min in this video?" | "Rectal stump." |
| Object Aggregation | Object Identification | frame | fo_class | "Which combination of foreign object classes is visible in this frame?" | "Clip, Silicone loop" |
| Object Attributes | Object Identification | frame | open_ended | "Is the external drain in this frame currently grasped by an instrument?" | "Yes." |
| Object Identification | Object Attributes | frame | open_ended | "Is the silicone loop visible in this frame currently grasped by an instrument?" | "No." |
| Spatial Localization (Camera) | Object Identification | frame | fo_class | "What class is the foreign object located in the bottom/left relative to the image center?" | "Clip" |
| Spatial Localization (Situs) | Object Attributes | frame | open_ended | "Which foreign object is not fully visible due to an instrument occlusion in this frame?" | "Silicone loop." |
| Temporal Localization | Spatial Localization (Camera) | frame | multiple_choice | "Where is the center of the Silicone loop located relative to the image center in this frame?" | "bottom/left" |
| Temporal Ordering | Spatial Localization (Situs) | frame | open_ended | "In which abdominal quadrant is the sponge located before being not visible for >1 min?" | "Lower left abdominal quadrant." |

*(Multi-Step Reasoning did not appear as a secondary tag anywhere in the
24,000-row sample — it only ever occurs as a primary capability in this
dataset.)*

### 2.9 Summary — how the pieces fit together

- One video can generate **both** frame-track and segment-track questions,
  and appears in exactly one QA split (train/test) but that split is
  independent of which `.lance` (train/val/test) partition its raw frames
  live in.
- Frame-track questions are answerable from a **single static image**, so
  they're restricted to instantaneous properties: what's visible, how many,
  where (in-frame position or anatomical region), and basic attributes.
- Segment-track questions require **temporal context** (a clip), which is
  why all time-based, ordering, aggregation-over-time, interaction, and
  multi-step-reasoning capabilities live exclusively there.
- Object Identification is the backbone skill: it's the single most common
  primary capability and by far the most common secondary capability,
  meaning nearly every question in the benchmark implicitly requires
  recognizing which foreign object(s) are involved even when that's not the
  main point of the question.
