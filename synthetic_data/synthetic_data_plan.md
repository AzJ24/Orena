# Synthetic data — concrete plan of action

Step-by-step build plan for the synthetic FO-insertion pipeline, naming the exact
**datasets** and **models** used at each step. Conceptual overview:
[synthetic_data.md](synthetic_data.md). Dataset details + licenses:
[datasets.md](datasets.md).

**Goal:** generate perfectly-labeled (image + question + answer) examples of known
foreign objects in unseen-procedure frames — priority on the two missing classes
**Mesh** (5 samples) and **Absorbable Hemostatic Agent** (0 samples).

**All models below are public/open-weight** (required by the FRAME resource rule) and
run on one H100/H200.

**Two locations — data vs. code:**

**Data only** → `/projects/datasets_ML/orena/synthetic/` (images/frames, QA pairs, and
per-frame metadata — nothing else):
```
synthetic/
  backgrounds/        # confirmed-empty frames, by source dataset
  crops/              # extracted real FO cut-outs (PNG+alpha), by FO class
  generated/          # final composited/edited images
  labels/             # QA pairs (FOCUS chat-jsonl), one entry per frame
  metadata/           # per-frame provenance + verifier accept/reject records
  disclosure.md       # datasets + models used, for challenge submission
```

**Code and everything else** → `/home/ajenane/orena/synthetic_data/` (in-repo, where
these docs live):
```
synthetic_data/
  *.md                # these planning docs
  scripts/            # pipeline code (extract, filter, crop, insert, verify, author)
  configs/            # per-phase / per-class config
  train/              # training entrypoints + configs for the mix-in runs
  logs/               # run logs
  runs/               # experiment outputs, eval results
```
Scripts take the data dir as a config/CLI arg (e.g. `--data-root
/projects/datasets_ML/orena/synthetic`), so no absolute paths are hard-coded.

---

## Phase 0 — Setup

- **Acquire datasets** (see [datasets.md](datasets.md) access column) and log each in
  `synthetic/disclosure.md` (name, version, license, URL, date).
  - OOD backgrounds: **DSAD** (open, start here), **AutoLaparo**, **SAR-RARP50**,
    **MultiBypass140**, **LapGyn4**, **SLAM**.
  - In-domain crops/backgrounds: **Cholec80/CholecT50/CholecTrack20**, **Endoscapes**.
  - Own data: **HeiCo + LapChole** (FOCUS) — source of common-FO crops.
- **Install models** (weights bundled locally, no inference-time internet):
  - **SAM2** (Segment Anything 2) — object cut-out + mask generation.
  - **GroundingDINO** (open-vocabulary detector, text-prompted) — emptiness filter +
    count verifier. (Alt: **YOLO-World**.)
  - **Qwen-Image-Edit** (Apache-2.0) — generative insertion / inpainting.
  - **Qwen3-VL** (open) — semantic/realism verifier ("does this look like a real
    surgical frame with exactly N X?").
  - `ffmpeg` / `decord` — video → frame extraction.

---

## Phase 1 — Build the confirmed-empty background pool

**Datasets:** DSAD, AutoLaparo, SAR-RARP50, MultiBypass140, LapGyn4, SLAM (+ in-domain).
**Models:** ffmpeg/decord (extract), GroundingDINO (emptiness filter).

1. Extract frames from each background dataset (stride tuned per dataset; images-only
   sets like DSAD/LapGyn4 skip this).
2. Run **GroundingDINO** over each frame with the FO vocabulary as text prompts
   (`sponge, clip, needle, specimen bag, drain, gallstone, mesh, hemostatic material…`).
   **Keep only frames it reports as FO-free** → these become confirmed-empty backgrounds.
   Instruments/blood/smoke are allowed (not FOs) → realistic clutter retained.
3. Human spot-check a sample per dataset to validate the filter.
4. Save to `synthetic/backgrounds/<dataset>/`. A subset kept untouched =
   **guaranteed negatives** (count 0 / "present? no").

---

## Phase 2 — Build the real object-crop library

**Models:** GroundingDINO (locate) + SAM2 (cut out).

Per FO class, obtain clean cut-outs with alpha masks:

- **Common FOs** (Clip, Sponge, Specimen Bag, Gallstone, Specimen, Needle, Silicone
  Loop, External Drain) —
  **Datasets:** own **HeiCo/LapChole** frames (already known to contain them), plus
  **CholecT50** (triplet/bbox labels pinpoint clip/bag/gallstone frames) and
  **CholecTrack20** (tool-tracking boxes).
  Locate with GroundingDINO/existing labels → cut with **SAM2** → store in
  `synthetic/crops/<class>/`.
- **Mesh** — **Dataset: SLAM** (abdominal wall hernia-repair clips). Pull frames from
  hernia clips, cut the mesh patch with **SAM2**. Only public real-pixel source.
- **Absorbable Hemostatic Agent** — **no crop source.** Harvest a few weak exemplars
  from **MultiBypass140** bleeding/IAE frames for reference/verification only; real
  insertion is generative (Phase 3B).

---

## Phase 3 — Insert objects into empty backgrounds

Two routes; use per class as noted.

### 3A — Composite real cut-outs (preferred where crops exist)
**Inputs:** `backgrounds/` + `crops/`. **Models:** image-harmonization
(**libcom** / **PCT-Net** / **Harmonizer**) for lighting/color match; Poisson blending
for seams.
- Place 0–3 known instances of a class at plausible locations/scales → harmonize.
- Use for: **all common FOs** and **Mesh** (real SLAM crops).
- Count is exact by construction (empty bg + N placed).

### 3B — Generative insertion (where no crop exists / for variety)
**Model:** **Qwen-Image-Edit** (mask a region → "add white/pale-yellow frizzy
hemostatic material on this bleeding tissue" / "add a surgical mesh patch here").
- Use for: **Absorbable Hemostatic Agent** (only route), and as augmentation variety
  for mesh/common FOs.
- Higher realism risk → leans hardest on Phase 4.

Save all outputs to `synthetic/generated/` with provenance (bg source, class, count,
method, placement).

---

## Phase 4 — Verify (quality gate)

**Models:** GroundingDINO/YOLO-World (re-detect + count) + Qwen3-VL (realism/semantics).

For every generated image:
1. Re-detect the FO → **accept only if detected count == intended count**.
2. **Qwen3-VL** check: "Is this a realistic surgical frame? Does it contain exactly N
   <class> and nothing anomalous?" → reject fakes/artifacts.
3. Log accept/reject + reasons to `synthetic/verify/`. Reject → discard.

This is what makes labels trustworthy — weighted hardest for **hemostatic agent** and
generative outputs, which have no real distribution to fall back on.

---

## Phase 5 — Author QA pairs (FOCUS formats)

**Model:** optional local **Qwen3** for phrasing variety (formats stay exact).

From the known contents (empty bg + verified insertions), emit QA in the exact scored
formats (`FO_definitions.txt` classes; `focus.data.formats`):
- `binary`: "Is a <class> present?" → `yes`/`no`
- `number`: "How many <class>?" → bare integer (incl. 0 from untouched negatives)
- `fo_class`: identification → exact registered class name
- situs/spatial where placement is known.
Write chat-jsonl matching `orena_sft/build_frame_sft_dataset.py` schema →
`synthetic/labels/`.

---

## Phase 6 — Mix, train, evaluate

**Model:** existing SFT stack in `orena_sft/` (Qwen VLM).
1. Mix synthetic in at **10–20%** with real FOCUS train.
2. Retrain (LoRA) with vs. without synthetic.
3. Evaluate on the **OOD proxy** (heico→lapchole transfer; sigma-resection held-out)
   via the FOCUS `Evaluator`. Track per-class accuracy on **Mesh** and **Hemostatic
   Agent** specifically.
4. **Keep synthetic only if OOD / rare-class accuracy improves.**

---

## Phase 7 — Iterate and scale

- Start small: **clips on DSAD** (crop route) + **hemostatic agent generative** on
  MultiBypass140 bleeding frames — a few dozen each, eyeball outputs.
- Scale the combinations that move the OOD metric; stop where the curve flattens
  (diversity-bound, not compute-bound — see [synthetic_data.md](synthetic_data.md)).
- Prioritize the two missing classes (Mesh via SLAM, Hemostatic Agent via generative),
  since they have ~0 training signal and the largest expected relative gain.

---

## Model + dataset summary

| Step | Datasets | Models |
|---|---|---|
| 0 Setup | all (acquire + disclose) | SAM2, GroundingDINO, Qwen-Image-Edit, Qwen3-VL |
| 1 Backgrounds | DSAD, AutoLaparo, SAR-RARP50, MultiBypass140, LapGyn4, SLAM | ffmpeg/decord, GroundingDINO |
| 2 Crops | HeiCo/LapChole, CholecT50/Track20; **SLAM (mesh)** | GroundingDINO, SAM2 |
| 3A Composite | backgrounds + crops | libcom/PCT-Net/Harmonizer |
| 3B Generative | (+ MultiBypass140 exemplars for hemostatic) | Qwen-Image-Edit |
| 4 Verify | — | GroundingDINO/YOLO-World, Qwen3-VL |
| 5 QA authoring | — | Qwen3 (optional) |
| 6 Train/eval | synthetic + FOCUS HeiCo/LapChole | Qwen VLM (orena_sft) |
