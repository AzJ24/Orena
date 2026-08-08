# Frame-track split (v2)

Regenerate and verify:

    .venv/bin/python orena_sft/build_frame_sft_dataset.py \
        --datasets heico lapchole \
        --out-dir orena_sft/sft_export/combined_v2 \
        --manifest orena_sft/split_manifest_v2.json

    .venv/bin/python orena_sft/verify_split_manifest.py

Regeneration is deterministic: rebuilding from scratch reproduces
`split_manifest_v2.json` byte-for-byte (video assignment AND the exact qID set
of every split, hashed).

## Why v1 was replaced

v1 stratified on `procedure_type` alone and held out whole videos with no row
cap. heico videos carry 400 questions each against lapchole's ~80, and heico has
only 20 train videos, so `round(10 * 0.12) = 1` video per procedure:

| | v1 | v2 |
|---|---|---|
| eval videos | 11 | 15 |
| **effective videos (Kish)** | **6.1** | **15.0** |
| largest single video | 26.3% of eval rows | 6.7% |
| top-2 videos | 52.5% of eval rows | 13.3% |
| train/eval qID leak | 2 rows | 0 |

An eval set whose effective sample size is 6 videos cannot separate a real
2-point effect from video-level noise, and every checkpoint selected on it
inherited that noise.

## What v2 changes

- Strata are `(source_dataset, procedure_type)`, not `procedure_type`.
- `--min-eval-videos-per-stratum 3` (was effectively 1 for heico).
- `--max-eval-rows-per-video 80` subsamples dense videos, preserving each
  video's own format proportions via largest-remainder allocation.
- Explicit `(source_dataset, qID)` de-duplication against train.

Eval format mix now tracks test closely (eval vs test): fo_class 44.7/42.8,
number 31.8/33.5, binary 9.2/11.6, open_ended 9.3/8.9, multiple_choice 5.1/3.2.

Cost: train drops 12225 -> 10625 rows (-13%) because more whole videos are held
out. That is the price of an eval set that can actually resolve an effect.

## Known, deliberate limitations

- **`Gallstone` is in test but not eval.** It occurs in only 3 of 92 videos; a
  video-level split cannot cover it without deleting a third of the training
  signal for an already-rare class. Keeping all 3 in train is the better trade.
- **heico eval cannot proxy heico test.** heico/train is Proctocolectomy +
  Rectal Resection; heico/test is 100% Sigmoid Resection, a procedure that
  appears nowhere in train. No split of train can measure that shift, so heico
  eval measures held-out-video generalisation only. Treat heico test as a true
  OOD probe and do not tune against it.
- Eval loss remains a poor selection signal in this project (eval_loss and test
  accuracy have dissociated repeatedly). v2 makes the eval set trustworthy; it
  does not make token-mean cross-entropy the right thing to select on.
