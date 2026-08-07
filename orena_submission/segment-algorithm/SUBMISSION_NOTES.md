# SEGMENT-track submission — build notes

Model: **`segment-27b-alldata-n80-650-20260730-merged`** — Qwen3.6-27B with a
LoRA r=8 SFT adapter merged in, trained on all segment data at 80 frames/clip.

## What changed from the template

| File | Change |
|:--|:--|
| `resources/segment_model/` | added — the merged checkpoint, re-sharded to ~8 GB (see below) |
| `resources/dummy_weights.pt`, `resources/model.py` | removed — no custom `nn.Module`; the model loads via `Qwen3_5ForConditionalGeneration.from_pretrained()` |
| `resources/prompts.py` | added — byte-for-byte copy of `orena_sft/prompts.py`, so the container prompts exactly as training did |
| `Dockerfile` base image | **unchanged from the template** (`2.5.1-cuda12.4`), after an earlier draft wrongly moved it to `2.12.0-cuda12.6` |
| `resources/clip_frames.py` | added — rebuilds training's 80-frame input and its absolute-timestamp metadata from the platform's trimmed 5 fps clip |
| `inference.py` | rewritten from the dummy CNN: load once, warm up, then per question decode → 80 frames → greedy `max_new_tokens=32` → `extract_answer`, with per-question `try/except` and a batch-budget guard |
| `requirements.txt` | pinned to the exact stack the checkpoint was evaluated with, plus `flash-linear-attention` |
| `Dockerfile` | base image kept; split into a `deps` stage (so the stack can be preflighted without the weights) and the image; weights split across two `COPY` layers; build-time assertion via `verify_env.py` |
| `reshard_weights.py` | added — regenerates `resources/segment_model/` from the merged checkpoint |
| `test/verify_timeline.py` | added — proves the prompt reconstruction matches training |
| `do_build_podman.sh`, `do_save_podman.sh`, `test_apptainer.slurm`, `test_native.slurm` | added — this cluster has no Docker; see "Building here" |

The template's `do_build.sh` / `do_save.sh` / `do_test_run.sh` are left untouched for
anyone building on a machine that does have Docker.

## The one real design problem: the clip timeline

This is the thing most likely to silently destroy accuracy, so it is worth stating
plainly.

**Training** never decoded a video. `segment_track/clip_sampling.py` read
pre-extracted JPEGs of the *source* procedure video, took 80 indices spanning
`[start_time, end_time]`, and passed the processor **absolute** frame indices plus
the source's real fps. That is what makes Qwen3-VL render `<1234.5 seconds>` markers
on the source timeline — and the system prompt explicitly tells the model those
markers "count from the START OF THE VIDEO, not from the start of this clip", then
asks it to read `time` answers off them. `time` is **38.5%** of segment training data.

**The platform** hands over a clip already trimmed to the window, at 5 fps, whose own
timeline starts at 0. Decoding it naively — as the template's dummy does — would
produce markers running from `<0.0 seconds>`, and every `time` answer would be wrong
by `start_time`, i.e. by minutes.

`resources/clip_frames.py` restores it: marker times come from
`linspace(start_time, end_time, 80)`, expressed against a fixed 25 fps grid, and the
pixels are the nearest frame the 5 fps clip actually holds for each instant. This is
sound because `Qwen3VLProcessor._calculate_timestamps` derives markers from
`frames_indices / fps` and nothing else (`total_num_frames` is read only on the
`do_sample_frames=True` path, which is disabled).

**Verified** by `test/verify_timeline.py` over all 6254 exported test records:

```
[1] marker times vs. training
    25 fps sources (4000 records): worst drift    0.0 ms
    30 fps sources (2254 records): worst drift   30.0 ms
[2] rendered prompt text vs. training
    25 fps sources: 15/15 prompts byte-identical to training
```

Training snapped markers onto the source's integer frame grid — 25 fps for heico,
30 fps for lapchole — and a `Request` carries no frame rate, so one grid has to be
assumed. 25 fps reproduces heico byte-for-byte; lapchole markers land at most half a
frame off (30 ms measured). `focus.data.formats.Time` accepts an answer within
**5 s**, so that residual is 0.6% of the scoring tolerance.

### Known residual gap (not fixable from the given inputs)

Below ~16 s of window the 5 fps clip holds fewer than 80 distinct frames, so
neighbouring samples repeat: **4.1%** of test rows (259/6254), against 0.3% at
training time where the source ran at 25–30 fps. Those questions see a temporally
coarser clip than the model was trained on. Nothing in `/input` can recover frames
the clip does not contain.

## Package/versions — and why each is pinned

**Base image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` — the template's own,
unchanged.** It is also what the FRAME submission shipped on and the platform ran
successfully, which makes it the only base image with evidence behind it.

`requirements.txt` is FRAME's proven set plus the one thing SEGMENT adds:

| package | why |
|:--|:--|
| `transformers==5.9.0` | `Qwen3_5ForConditionalGeneration` + the Qwen3-VL processor whose timestamp rendering `clip_frames.py` is written against |
| `accelerate==1.14.0` | lets `from_pretrained()` stream 55 GB straight to the GPU instead of via host RAM |
| `orena-focus==0.3.5` | the FO class names in the system prompt come from this release's `FOType.names()` |
| `decord==0.6.0` | decodes the platform's clips — the one dependency SEGMENT adds over FRAME |
| `numpy>=1.23.0`, `pillow>=9.0` | direct imports, declared explicitly |

**torch is deliberately NOT pinned.** The base image tag is the single source of
truth for the torch/CUDA pair, and the Dockerfile asserts it at build time via
`verify_env.py`. Pinning torch — or letting a dependency drag one in — replaces the
base image's build with a PyPI wheel built for a different CUDA, which does not
raise: `torch.cuda.is_available()` returns False and inference runs on CPU.
Verified by dry-run and by a real serial install: this requirement set leaves the
base image's torch 2.5.1+cu124 **untouched**.

### `flash-linear-attention` is deliberately ABSENT

48 of the model's 64 layers are linear-attention (Gated DeltaNet), and the offline
evaluation used fla's fused kernels — so including it looks obviously right. It is
not:

- fla needs `triton >= 3.3`; the base image's torch 2.5.1 pins `triton 3.1.0`. On
  that combination fla does not fall back, it **raises while transformers imports
  it** — `TypeError: Autotuner.__init__() got an unexpected keyword argument
  'do_bench'` — killing the run at model construction. Observed: a 20-question batch
  died after 16.6 s with exit code 1.
- Installing it therefore forces a newer base image, which is the whole reason an
  earlier draft of this submission moved off the template's.
- And it buys nothing worth that. Measured on this checkpoint over the same 20 real
  TEST questions: **4.09 s/question** on the pure-torch path vs **2.87 s** with fused
  kernels — against a **15 s** budget.

`verify_env.py` asserts `flash-linear-attention` and `fla-core` are absent, so a
future "helpful" addition fails the build rather than the submission. The
*"The fast path is not available…"* warning at load is **expected and correct** here;
FRAME shipped with the same warning and passed.

## Image layout — the 50 GB layer limit

A single image layer must not exceed 50 GB and each `COPY` is one layer. As merged,
the checkpoint's larger shard was **49.83 GB** — under the limit by 0.35%, which is
not a margin worth risking a 55 GB upload on. `reshard_weights.py` streams it into
seven ~8 GB shards (verified: all 1184 tensors present, spot-checked byte-identical),
and the Dockerfile's two globbed `COPY`s make layers of ~32 GB and ~23 GB.

## Latency

The budget is `120 s + 15 s x batch_size`, pooled, and a >20% overrun forfeits the
whole batch. Offline eval of the sibling 27B/80-frame checkpoint over 6254 questions:

| | mean | median | p90 | max |
|:--|--:|--:|--:|--:|
| prep (decode + encode) | 0.57 s | 0.58 s | 0.65 s | 0.89 s |
| generate | 2.31 s | 2.23 s | 2.48 s | 3.60 s |
| **total** | **2.89 s** | **2.85 s** | **3.10 s** | **4.32 s** |

### Measured on the SHIPPING stack (gpu38 / H200, job 13207)

`inference.py` on **torch 2.5.1 + CUDA 12.4** — the image's actual base stack, via
`venv-cu124` — against 20 real FOCUS TEST questions, VRAM capped to the platform's
80 GiB:

```
torch 2.5.1+cu124 | CUDA 12.4 | available True
[sim] VRAM capped to 80 GiB of 139.8 GiB

exit code 0     responses 20/20     empty 0
WALL CLOCK 106.0s / BUDGET 420s     (25.2% used)
per-question: mean 4.09s   max 5.41s   over 15s: 0
>>> PASS <<<
```

- **25.2% of budget used**, worst question 5.41 s against a 15 s allowance.
- **Peak VRAM 54.0 GiB** — fits the platform's 80 GiB H100 with ~26 GiB headroom.
- **20/20 answered, none empty**, every answer in a shape the parsers accept.
- **The `time` question was exact**: `pred='02:36:13' gold='02:36:13'` — an absolute
  source-video timestamp recovered from a clip that starts at zero. This is the
  clearest evidence the timeline reconstruction works on genuinely encoded 5 fps
  clips, not just in the unit check.

Deterministic accuracy on that sample was 7/12 (58.3%), consistent with the offline
eval; 20 questions is far too few to read as a score, and `open_ended` /
`multiple_choice` are LLM-judged on the platform so they are excluded here.

The batch is regenerated by `test/build_real_batch.py`, which re-encodes real TEST
windows to the platform's stated clip spec (H.264, exactly 5 fps, height <= 576,
keyframe every 5 s). It round-robins the answer formats: taking questions in dataset
order yields 20 `fo_class` rows and never exercises `time` at all.

### Setup is the part to watch — and the `.sif` numbers are NOT the platform's

Running the shipped `.sif` under apptainer reports an alarming setup cost, and it is
a measurement artefact. All four paths, same model:

| path | setup |
|:--|--:|
| native, warm page cache | 8.6 s |
| **uncompressed weights on local NVMe (job 13284)** — the platform's case | **13.1 s** |
| native, **cold** NFS read of raw safetensors | 184 s |
| `.sif` squashfs on **local NVMe** (job 13283) | 362 s |
| `.sif` squashfs over NFS (job 13282) | 404 s |

Moving the `.sif` to local NVMe — ~8x the bandwidth — bought only 10%, so setup is
**not I/O bound**. It is bound by squashfs decompression, at a rate identical on
both tracks: frame 18.8 GB / 127 s = 148 MB/s, segment 51 GB / 362 s = 141 MB/s.
Copying the 44 GB `.sif` onto that same NVMe took 19 s (~2.3 GB/s), so decompression
runs at ~1/16 of the disk.

**The platform does not pay this.** An Apptainer SIF *is* a squashfs; a Docker/OCI
image is tar layers extracted to overlayfs — ordinary uncompressed files — and the
image is pulled before the process starts, which the template states is not charged
to the budget.

`test_uncompressed_load.slurm` (job 13284) measured that case directly — the same
51 GB of weights, uncompressed, on the same local NVMe — and settles it:

```
staging 51 GB to local NVMe      14.45 s   (not charged; mimics the image pull)
Model loaded and set to eval mode  9.76 s
Setup complete in                 13.12 s
20 questions, mean 4.24 s each
>>> PASS: 107.5s within 420s budget <<<   (25.6% used)
```

**362 s -> 9.8 s.** Squashfs decompression was the entire cost.

One caveat on that figure: 51 GB in 9.76 s is ~5.2 GB/s, above the ~2.3 GB/s the
`.sif` copy sustained, so the `--mem=48G` cgroup did not fully defeat the page cache
and part of the read was served warm. A fully cold NVMe read would be nearer 25-30 s.
Either number is immaterial — at 4.24 s/question, `setup + 4.24B <= 120 + 15B` holds
with wide margin for every batch size down to B=1.

The corollary matters for reading any apptainer result on either track: they are
pessimistic upper bounds. FRAME passed the platform with an apptainer-measured
127-141 s setup.

**What the 20-question `.sif` run actually showed (job 13283):** 20/20 answered,
**0 skipped, 0 empty**, mean 4.19 s/question, peak VRAM 55.1 GiB — 465.3 s against a
420 s budget. That is 10.8% over, i.e. a *proportional* forfeit, still inside the
20% whole-batch cliff, in the pessimistic squashfs scenario. The 3-question run
(job 13282) failed harder only because a 3-question budget is 165 s, so a fixed
setup cost dominates.

`inference.py` still does a warm-up generation after loading, so any first-call cost
lands in setup rather than on question 1.

### The budget guard — two bugs, and what ships

**First version:** stopped generating past 85% of the *nominal* budget. Setup alone
exceeded a 3-question allowance, so it skipped all three and wrote empty answers,
turning a slow-but-scoring run into a guaranteed zero. Rewritten to target the real
cliff: overrunning the nominal budget forfeits a growing *share*, and only **+20%**
loses the batch, so a question is skipped only when starting it would cross
`1.2 x budget`, estimated from the running mean of questions already answered.

**Second version** (what is in the shipped image) still had a flaw the apptainer run
exposed: once setup had *already* passed the cliff, every question was skipped —
but skipping cannot refund time already spent, so it guaranteed zeros where
answering might still have scored. `inference.py` now only skips while `t0` is still
under the cliff, and logs `past the ... cliff already — answering anyway` otherwise.

**This fix is NOT in the built image, deliberately.** Old and new differ only when
`t0 > hard_deadline`, and past that point the whole batch is forfeited either way —
so the fix cannot convert a lost question into a scored one. Rebuilding costs ~2-3 h
of re-export plus a 44 GB re-upload for zero scoring benefit. It is kept in source
for any future build. Note that on the realistic 20-question run the shipped guard
did **not** fire at all (0 skipped).

## Verification status — what is and is not proven

Everything below the line was run; nothing above it has been, because no node
reachable from this account has a container runtime. Keep the distinction: the
inference path is measured, the packaging is not.

**Verified** (gpu38 / H200, `venv-cu124` = the image's exact stack, torch 2.5.1+cu124):

| check | result |
|:--|:--|
| `verify_env.py --build-time --gpu` | all OK — CUDA 12.4, `is_available()` True, sm_90, bf16 matmul |
| `test/verify_timeline.py`, all 6254 records | PASS — 25 fps byte-identical, 30 fps <= 30 ms vs a 5 s tolerance |
| template's own 3-question sample batch | exit 0, `'No' / '0' / 'No'`, ~3.1 s each |
| 20 real TEST questions, VRAM capped to 80 GiB | exit 0, 20/20, 106 s of 420 s, peak 54.0 GiB, `time` answer exact |
| `reshard_weights.py` output | 1184/1184 tensors, spot-checked byte-identical |
| Dockerfile `COPY` globs | 4 + 3 = all 7 shards -> ~30 GB and ~22 GB layers |
| build context | 52 GB (`test/` excluded by `.dockerignore`) |
| numpy 2.4.4 <-> torch 2.5.1 | interop OK |

**Not verified — all build-side:**

1. **The Dockerfile has never been built.** Base image pull, `apt-get`, `pip install
   --user`, the multi-stage `--target deps`, and real layer sizes are unexercised.
2. **`pip install --user` into `/home/user/.local`** — dependency resolution was
   tested in a venv, not the image's `--user` layout.
3. **`verify_env.py` running inside `docker build`** — the assertion is proven, its
   invocation in a build is not.
4. **The `.sif` under apptainer** — needs the build to exist.
5. **The budget-guard skip branch** has not fired since it was fixed. It only
   triggers near the 1.2x cliff and the batch runs at 25% of budget, so it should
   not — but it is untested code.
6. **Cold-start load from image layers** — 8.7 s warm / 184 s cold over NFS; from
   local image storage, unmeasured.

`preflight_deps.sh` exists for items 1-3 and is the reason to run it first.

## Guarding against the FRAME-track silent failure

On the frame track, `torch==2.9.1` + CUDA 13.0 were pinned in `requirements.txt`
against a CUDA 12.4 base image. Nothing raised — a torch built for the wrong CUDA
just reports `torch.cuda.is_available() == False`, runs a 27B model on CPU, and the
loss only shows up after the image is built, uploaded and scored.

Three things here are designed against exactly that:

**1. torch is not pinned, and nothing drags one in.** The base image tag is the
single source of truth. Verified two ways rather than argued from metadata:
`pip install --dry-run -r requirements.txt` against a clean torch 2.5.1+cu124 venv
plans **no** torch/torchvision/triton change, and a real serial install leaves
`torch 2.5.1+cu124` in place.

> Caveat worth recording, because it cost hours: an earlier test appeared to show
> the template base silently upgrading to `torch 2.13.0+cu130`. That was an
> artefact of running two `pip` processes concurrently against one venv — torch had
> landed but torchvision had not, so `orena-focus` pulled torchvision 0.28.0, which
> hard-pins `torch==2.13.0`. **Install serially**, or you will reproduce it. The
> mechanism is real even if the original conclusion was not: an incomplete
> environment is exactly how the wrong torch gets in.

**2. `verify_env.py` asserts it, and the build fails if it is wrong.** It runs
inside `docker build` (no GPU needed) and checks the CUDA major, the exact torch
version, that `flash-linear-attention` is **absent**, that the FO class list still
matches training — and, specifically, **where torch was imported from**: a
pip-installed torch shadowing the base image's lands in `/home/user/.local/`, which
is precisely the frame-track fingerprint.

**3. `preflight_deps.sh` runs it before any weights are copied.** The Dockerfile is
split into a `deps` stage and the final image, so `--target deps` builds a ~4 GB
image from a two-file context in minutes. **Run this first** — it is the step that
would have caught the frame-track failure for the price of a coffee rather than a
build, an upload and a scoring round.

The check is live, not decorative: pointed at this repo's `venv3.12` (which is
cu130) it fails as intended, and at `venv-cu124` it passes:

```
[FAIL] torch CUDA major is 12 — built for CUDA 13.0 — this is the silent-CPU-fallback bug
[OK  ] torch version is 2.5.1 — got 2.5.1
```

What preflight cannot prove is that CUDA works on a real device — there is no GPU on
the build node. `test_apptainer.slurm` closes that gap with `verify_env.py --gpu`.

## Simulating the platform in testing

`test_apptainer.slurm` runs the shipped `.sif` the way the platform runs the image:

| platform condition | how it is reproduced |
|:--|:--|
| `/input` read-only, `/output` writable, `/tmp` a volume | the same three binds |
| networking disabled | `--net --network none` (probed first; some `apptainer.conf` forbid unprivileged netns, in which case it warns rather than fails) |
| 1x H100, 80 GiB | `--gres=gpu:h200:1` on gpu38 for sm_90, **plus** `set_per_process_memory_fraction` capping VRAM to 80 GiB so an allocation that would OOM on the platform OOMs here instead of fitting in the H200's 143 GiB |
| clock starts at process start | the whole run is timed and checked against `120 + B x 15`, including the 20% forfeit cliff |
| answers | `answer.json` verified for count, qID match and empties |

Three workarounds are **apptainer-only** and would be wrong to read as platform
problems — Docker needs none of them:

- `--no-home` plus `HOME=/home/user`: apptainer mounts the *host* home over the
  container's, hiding the `pip --user` packages in `/home/user/.local`.
- `--pwd /opt/app`: apptainer ignores the image's `WORKDIR`.
- `apptainer exec … python …` rather than `run`: a consequence of the two above, so
  the `ENTRYPOINT` itself is only exercised by the platform.

## Building here (no Docker on this cluster)

Per the cluster tutorial: build with rootless **Podman** on `cpu34`, run with
**Apptainer** on `gpu38`. The upload artifact is a `docker-archive` `.tar.gz`, which
`podman save --format docker-archive` produces identically to `docker save`; the
`.sif` is only for local GPU testing and is **not** submitted.

```bash
# 0. (once) regenerate the weights if the checkpoint changes
venv3.12/bin/python orena_submission/segment-algorithm/reshard_weights.py \
    --src segment_track/checkpoints/segment-27b-alldata-n80-650-20260730-merged \
    --dst orena_submission/segment-algorithm/resources/segment_model

# 1. logic check on a GPU, no container   (from the repo root)
sbatch orena_submission/segment-algorithm/test_native.slurm

# 2. PREFLIGHT the dependency stack       (on cpu34, ~minutes, ~4 GB)
ssh cpu34
cd ~/orena/orena_submission/segment-algorithm
./preflight_deps.sh        # <- do not skip; this is the frame-track guard

# 3. build + package                       (on cpu34, ~an hour)
./do_build_podman.sh
./do_save_podman.sh        # -> segment-algorithm_<stamp>.tar.gz  +  ~/images/segment-algorithm-1.0.sif

# 4. run the real image under platform conditions (from the repo root)
sbatch orena_submission/segment-algorithm/test_apptainer.slurm

# 5. keep /localbuild tidy on cpu34
podman rmi segment-deps-check segment-algorithm:1.0   # once the .sif is verified
podman system prune
```

`ssh cpu34` currently fails with *Permission denied (publickey)* from this account —
that needs sorting before step 2.

### Disk

The weights exist twice on `/home` (merged checkpoint + re-shard, ~110 GB). The
`.tar.gz` adds ~55 GB and the `.sif` another ~55 GB. `do_save_podman.sh` keeps its
intermediates on local disk, but check `df -h ~` before starting, and consider
deleting the original merged checkpoint once the image is verified.

## Submitting

1. [orena-focus-challenge.org](https://orena-focus-challenge.org/) → the **segment**
   track subdomain → **Submit** → pre-evaluation phase → **Manage your algorithms** →
   create an algorithm (once per track).
2. **Containers → Upload a Container** → the `.tar.gz` from step 2. Wait for
   *"Container image import completed"* before submitting — a 55 GB image takes a
   while, and submitting mid-import fails.
3. **Submit** tab → pick the algorithm → Save. Results take time; each submission
   fans out into many per-batch jobs. Up to 10 submissions per team in pre-evaluation.
