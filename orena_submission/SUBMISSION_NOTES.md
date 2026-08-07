# FRAME-track submission — build notes

## Where things are

- **Submission folder:** `orena_submission/frame-algorithm/` (cloned from the
  [orena-focus-submission-template](https://github.com/IMSY-DKFZ/orena-focus-submission-template)
  `frame-algorithm/` scaffold — only the FRAME track was built, not SEGMENT/PROCEDURE).
- **Docker build context:** `orena_submission/frame-algorithm/` itself — that's what
  `do_build.sh` / `do_save.sh` pass to `docker build`. Nothing has been built yet
  (no Docker/Singularity/Apptainer available on this cluster's nodes, including gpu38).
- **Model weights baked into the image:** `orena_submission/frame-algorithm/resources/frame_model/`
  — a hardlinked copy of `models/combined-all-9b-8r-direct-ddp-merged/` (Qwen3.5-9B +
  LoRA-SFT checkpoint-938, merged, ~18.8 GB `model.safetensors`).

## What was changed from the template

| File | Change |
|:--|:--|
| `resources/frame_model/` | added — the merged checkpoint (replaces `resources/dummy_weights.pt`, which was removed) |
| `resources/model.py` | removed — not needed, the model loads via `transformers.Qwen3_5ForConditionalGeneration.from_pretrained()` directly, no custom `nn.Module` |
| `resources/prompts.py` | added — the `direct`-style system prompt + `extract_answer` parser, copied byte-for-byte from `orena_sft/prompts.py` (verified identical via a diff script) so the container prompts the model exactly as it was trained/evaluated |
| `inference.py` | rewritten from the dummy CNN template to: load the merged model once, then per question build the same chat template `orena_sft/evaluate_qwen_frame.py` uses, generate greedily (`max_new_tokens=32`, matching the `direct` eval default), and parse with `extract_answer`. Per-question `try/except` so one bad frame doesn't cost the batch. |
| `requirements.txt` | pinned `orena-focus==0.3.4`, `transformers==5.9.0`, `torch==2.12.0`, `accelerate==1.14.0` to exactly match the training/eval venv — the base image's stock torch 2.5.1 is too old for `Qwen3_5ForConditionalGeneration` |
| `Dockerfile` | comments updated to reflect the real weights (no COPY-splitting needed — single file is under the 50 GB layer limit) |
| `resources/README.md` | updated to describe `frame_model/` + `prompts.py` instead of the dummy files |

## Known limitation (not fixed, by design)

The system prompt uses the **static** 10-class foreign-object registry baked into
`orena-focus==0.3.4`, not the per-run `FO_definitions.json` the platform provides at
inference. This matches exactly how the checkpoint was trained and evaluated. If the
test phase adds extra FO classes via that file, the model won't know about them —
that's a modeling gap, not a packaging bug, and changing it now would mean testing
against an untrained prompt.

## Validated so far (no Docker/GPU test yet)

- `resources/prompts.py`'s `build_system_prompt()` output is byte-identical to
  `orena_sft/prompts.py`'s `build_system_prompt(False, style="direct")`.
- `AutoProcessor.from_pretrained(resources/frame_model)` loads correctly and
  `apply_chat_template` + `processor(...)` produce the expected multimodal tensors
  against the template's sample frame (`test/input/interface_1/frames/q001.png`).
- **Not yet run:** a full generation pass, `do_test_run.sh`, or a real `docker build`.
  The merged model needs ~19 GB VRAM in bf16; the GPUs reachable from this shell
  (Quadro RTX 5000, 16 GB) are too small, and no container runtime is installed on
  gpu38 or the other cluster nodes checked so far.

## What's left before submitting

1. **Test it end to end.** Either:
   - run `./do_test_run.sh` on a machine with Docker and enough GPU memory (an
     RTX PRO 6000 / 97 GB node like gpu38 would work, but needs Docker installed
     there first — it currently isn't), or
   - run `inference.py`'s logic directly (no Docker) against
     `test/input/interface_1/` on a big-enough GPU, to sanity-check the model
     loads and produces reasonable answers before trusting a full Docker build.
2. **Build + save the image:**
   ```bash
   cd orena_submission/frame-algorithm
   ./do_build.sh      # docker build --platform=linux/amd64 -t frame-algorithm .
   ./do_test_run.sh   # build + run against the sample batch, writes test/output/interface_1/answer.json
   ./do_save.sh        # build + docker save -> frame-algorithm_<timestamp>.tar.gz (next to the scripts)
   ```
   This machine has no Docker — these need to run wherever Docker is available
   (a workstation, or a cluster node with Docker/nvidia-container-runtime installed).
3. **Register / create the algorithm** on the challenge platform, once per track:
   [orena-focus-challenge.org](https://orena-focus-challenge.org/) → your track's
   subdomain (e.g. `frame.orena-focus-challenge.org`) → **Submit** tab →
   pre-evaluation phase → **Manage your algorithms** → create a new algorithm.
4. **Upload the container:** in that algorithm, **Containers → Upload a Container**,
   upload the `*.tar.gz` from step 2. Wait for the platform to report
   "Container image import completed" before submitting — don't submit while it's
   still importing.
5. **Submit:** back in the **Submit** tab, pick your algorithm from the dropdown and
   click Save. Results take time to appear (each submission runs many
   per-batch jobs); don't resubmit just because nothing seems to be happening yet.
   Up to 10 submissions are available per team during pre-evaluation.
