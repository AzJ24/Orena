# Synthetic data pipeline — inserting known foreign objects into real surgery frames

Plain-language description of the plan for generating extra, perfectly-labeled
training examples for the FOCUS FRAME track. See [datasets.md](datasets.md) for the
external background datasets and their licenses.

## The core idea (in one sentence)

Take a **real** photo from some surgery, **paste a foreign object into it ourselves**,
and because *we* chose what to paste, we **already know the right answer** — so we get
a perfectly-labeled new training example for free.

Think of it like Photoshopping a sponge into a surgery picture. Since you're the one
who put exactly one sponge in, you know the answer to "how many sponges?" is 1. No
guessing, no manual labeling.

## Why we do it this way

Two problems we solve at once:
- The challenge test will show the model **surgeries it never saw in training**
  (e.g. hysterectomy, when we only trained on gallbladder + colon surgery).
- Some foreign objects (like a gallstone) barely appear in our training data.

So we need pictures of **known objects** sitting inside **unfamiliar surgery scenes**.
We can't get those from the challenge data. But we *can* build them: grab real
unfamiliar-surgery photos from other public datasets, and drop the objects in
ourselves. The foreign-object vocabulary is closed and fully listed in
`FO_definitions.txt`, so we never invent new object types — we only place known ones.

## The pipeline, step by step

**Step 1 — Collect *confirmed-empty* background photos.**
The "canvas": real frames from other surgeries (hysterectomy, prostatectomy, etc.)
from the public datasets in `datasets.md`. The *scene* is genuine — that's what makes
it valuable; only the object gets added.

We deliberately keep **only backgrounds that are confirmed to contain no foreign
objects**, so that after we add N objects, the total is *exactly* N — the label is
correct by construction, with no counting of anything pre-existing. How we confirm
emptiness depends on the source:
- **Challenge frames whose ground-truth answer says "no FO present"** → provably empty
  (the gold case). Downside: in-distribution, so useful for in-domain robustness but
  not the unseen-surgery goal.
- **External datasets** (hysterectomy, prostatectomy, DSAD…) have *no* FOCUS foreign-
  object labels, so we confirm emptiness by running the object-finder as a **rejection
  filter** — keep only frames where it detects zero foreign objects — and by favoring
  frame types unlikely to contain them (DSAD anatomy/dissection frames, early phases
  before any clip/suture). A human spot-check on a sample keeps the filter honest.

**Important — "empty of foreign objects" is not "empty scene."** Per `FO_definitions.txt`,
instruments (graspers, scissors, trocars, staplers, cameras) are **not** foreign
objects. So a frame with instruments, blood, or smoke but no sponge/clip/needle still
counts as empty — and gives us **realistic clutter for free**, which we want, since the
real test frames are cluttered, not pristine.

Two bonuses of starting empty:
- **Free, guaranteed-correct negatives** — an untouched empty frame is a perfect
  "foreign object present? → no" / "count → 0" example.
- **Full control over the count** — starting from zero, we can add 1, 2, or 3 *known*
  objects and the count stays exact, so we cover multi-object scenes without ever
  losing label certainty.

**Step 2 — Get the objects to insert.**
Two ways to obtain a "sponge" or "clip" to place:
- **Cut a real one out** of a challenge photo where it already appears, using **SAM2**
  (an "auto object outliner": it traces an object's exact outline so we can cut it out
  cleanly, like a sticker).
- **Or have an image-editing AI draw it in** (e.g. Qwen-Image-Edit): tell it
  "add a surgical sponge here" and it paints one.

Cutting a real one out usually looks more realistic; the AI is more flexible. We'll
likely use both.

**Step 3 — (folded into Step 1) Guarantee the background is empty.**
Because Step 1 already keeps only confirmed-empty backgrounds, there's nothing to
"measure" here: the baseline foreign-object count is known to be zero. The object-finder
is used as an **emptiness filter** on the background pool (reject any frame that already
shows a foreign object), *not* to tally pre-existing objects. This is what makes every
downstream label correct by construction.

**Step 4 — Insert the object.**
Paste the cut-out object into the background (blending edges and lighting so it doesn't
look like a sticker), or let the editing AI paint it in. We control **what** object,
**how many**, and roughly **where** — those three facts become our answer key.

**Step 5 — Verify (the quality gate — the important one).**
The paste or the AI can go wrong: the object might not show up, look fake, or end up
duplicated. So we run the **object finder again on the finished picture** and check:
- Is the object we wanted actually there?
- Is the count right?
- Does it look believable?

If any check fails, we **throw that picture away**. This is what keeps labels
trustworthy — without it we'd train on answers we merely *hope* are right, which is
worse than no data.

**Step 6 — Write the question and answer.**
We now know exactly what's in the picture (original contents from Step 3 + what we added
in Step 4, confirmed in Step 5). So we auto-write question–answer pairs — "Is a sponge
present? → yes", "How many clips? → 2" — in the **exact format the challenge grades**
(bare "yes"/"no", bare number, etc.), so scoring counts them correct.

**Step 7 — Train and check it actually helps.**
Mix the synthetic examples in with real ones (a modest fraction, ~10–20%), retrain, and
compare performance **with vs. without** the synthetic data on our unseen-surgery test.
Keep them only if the unseen-surgery score goes up. If it doesn't help, we stop or
adjust — we don't add data just because we can.

## The whole thing as a loop

```
real surgery photos
      │
   keep only CONFIRMED-EMPTY backgrounds   ← object finder as emptiness filter
      │                                       (instruments/blood OK, no FOs)
   insert N known objects                   ← SAM2 cut-out  OR  editing AI
      │                                       (total is exactly N; N can be 0 → negatives)
   check it came out right                  ← object finder again  (toss failures)
      │
   write the Q&A from what we know
      │
   add to training set → retrain → did unseen-surgery score improve?
```

## Where it lives

Everything goes under `/projects/datasets_ML/orena/synthetic/` — generated images,
their labels, and the verification records.

## Honest caveats

- **The verifier does a lot of work.** It's what makes labels trustworthy. It won't be
  perfect, so we eyeball a sample of outputs early to confirm it catches failures.
- **Realism matters.** A pasted object that looks fake can teach the model the *wrong*
  thing (it learns to spot our paste-marks instead of real objects). Blending + the
  verifier's "does it look believable" check guard against this.
- **It's an experiment, not a guarantee.** We prove it helps on the unseen-surgery test
  before scaling up. Start small (one object type, a few dozen images), look at them,
  measure, then decide.

## First step

Scaffold the `synthetic/` folder and build **Steps 1–5 for one object (clips)** on the
open DSAD dataset, so we can look at real generated examples before wiring in the Q&A
generation (Step 6) and training (Step 7).
