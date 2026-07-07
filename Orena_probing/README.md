# Orena_probing — Linear Probing of Vision Encoders

Compares how well different frozen pretrained vision encoders (CLIP, DINOv2,
BiomedCLIP) represent the HEICO surgical-video probing dataset, by training a
linear classifier on top of each encoder's frozen embeddings and comparing
accuracy / macro-F1 / balanced accuracy.

## Methodology

1. **Freeze encoders** — each pretrained model is loaded and frozen, no gradient updates.
2. **Feature extraction** — every image is passed through the encoder (`image -> encoder -> fixed embedding vector`) and the embeddings are cached to disk per model/task/split.
3. **Train a linear probe** — `embedding -> nn.Linear -> class prediction`. Only this one layer is trained, on top of the frozen embeddings. Early stopping is driven by a validation split carved out of train; the test split is never touched until the very end.
4. **Evaluation** — accuracy, macro-F1, and balanced accuracy (important given class imbalance) on the held-out test split.

This means there's no single "entry point" script — there are three pipeline
stages, run in order, each one's output on disk feeding the next:

```
extract_features.py   →   train_probe.py   →   plot_results.py
  (Steps 1 + 2)             (Steps 3 + 4)         (visualize)
```

## Directory structure

```
Orena_probing/
├── config.yaml              # encoders, tasks, paths, training hyperparams, wandb settings
├── extract_features.py      # CLI: encoder -> embeddings, cached to features/
├── train_probe.py           # CLI: embeddings -> trained linear probe -> results/probe_results.csv
├── plot_results.py          # CLI: results/probe_results.csv -> bar charts + heatmap PNGs
│
├── encoders/                # one frozen pretrained vision model each
│   ├── base.py               # BaseEncoder ABC: freezing, batching, encode()
│   ├── clip.py                # CLIPModel (openai/clip-vit-base-patch32), via transformers
│   ├── dino.py                # DINOv2 (facebook/dinov2-base), via transformers
│   ├── biomedClip.py          # BiomedCLIP (microsoft/BiomedCLIP-...), via open_clip
│   └── __init__.py            # ENCODER_REGISTRY + build_encoder(name, device)
│
├── data/                     # turns a HF dataset row into (images, labels) arrays
│   ├── dataset.py              # load_task(), task_input_type(), get_label_info()
│   └── preprocess.py           # prepare_task(), split_train_val()
│
├── probes/
│   ├── linear_probe.py        # LinearProbe (a single nn.Linear) + train_linear_probe()
│   └── checkpoints/           # generated: <encoder>/<task>.pt trained probe weights
│
├── features/                 # generated: <encoder>/<task>/{train,test}.npz cached embeddings
├── results/
│   ├── probe_results.csv      # generated: one row per (encoder, task) with all metrics
│   └── plots/                 # generated: accuracy.png, macro_f1.png, balanced_accuracy.png, macro_f1_heatmap.png
└── wandb/                    # generated: local W&B run cache (synced to wandb.ai)
```

`features/`, `results/`, `probes/checkpoints/`, and `wandb/` are all generated
by running the scripts — none of them need to exist beforehand.

## Setup

From the `orena/` repo root:

```bash
uv sync
source .venv/bin/activate
```

This installs everything in `pyproject.toml`, including `torch`, `transformers`,
`open_clip_torch`, `scikit-learn`, and `wandb`. Make sure you're logged into
W&B once (`wandb login`) if you want `mode: online` (the default).

## Configuration (`config.yaml`)

```yaml
data:
  source: local                 # "local" (probing_export/) or "hub" (HF Hub)
  local_dir: ../probing_export
  hub_repo: Machine-Learning-Oncology/orena_probing

encoders:
  clip:       { model_id: openai/clip-vit-base-patch32, device: cuda:0 }
  dino:       { model_id: facebook/dinov2-base,          device: cuda:1 }
  biomedclip: { model_id: "hf-hub:microsoft/BiomedCLIP-...", device: cuda:2 }

tasks:
  single_frame: [...]   # wired up end-to-end
  pair: [...]            # on hold — see "Tasks on hold" below
  window: [...]          # on hold

paths:
  features_dir: features
  results_dir: results
  checkpoints_dir: probes/checkpoints

training:
  batch_size: 64
  epochs: 150     # upper bound only — patience below decides the real stopping point
  lr: 0.001
  weight_decay: 0.0001
  patience: 10    # epochs without val-loss improvement before stopping
  val_split: 0.2  # fraction of train.npz carved out as a validation set
  seed: 42
  device: cuda

wandb:
  project: orena-probing
  entity: null    # null = your default wandb entity
  mode: online    # online, offline, or disabled
```

Each encoder has its **own GPU** in `device:` — this lets you extract/train
all three encoders in one process without them fighting for the same GPU.
Any `--device` CLI flag overrides this for every encoder in that run.

## Running the pipeline

### 1. Extract features (Steps 1+2)

```bash
python extract_features.py --encoders clip dino biomedclip --tasks all --split train test
```
- `--encoders`: any subset of the names in `config.yaml`'s `encoders:` (default: all of them).
- `--tasks`: `all`, or specific task names (e.g. `--tasks grasped_by_instrument`). Only `single_frame` tasks are implemented — pair/window tasks print a skip message.
- `--split`: `train`, `test`, or both (default: both).
- `--device`: override every encoder's device for this run.

Writes `features/<encoder>/<task>/<split>.npz` (embeddings, labels, qIDs, label metadata).

### 2. Train + evaluate linear probes (Steps 3+4)

```bash
python train_probe.py --encoders clip dino biomedclip --tasks all
```
Requires `features/` to already exist (run step 1 first). For each (encoder, task):
- splits the cached train embeddings into train/val (stratified where possible),
- trains a `LinearProbe`, with early stopping on validation loss,
- evaluates the best checkpoint on the test split,
- logs per-epoch train/val loss + final metrics to W&B,
- saves the trained probe to `probes/checkpoints/<encoder>/<task>.pt`,
- appends a row to `results/probe_results.csv`.

Useful flags:
- `--encoders`, `--tasks`: same as above.
- `--device`: override every encoder's device for this run.
- `--wandb-mode {online,offline,disabled}`: override `config.yaml`'s wandb mode (e.g. `disabled` for a quick local run with no network calls).

### 3. Plot results

```bash
python plot_results.py
```
Reads `results/probe_results.csv`, writes grouped bar charts (one per metric)
comparing encoders across tasks, plus a macro-F1 heatmap, to `results/plots/`.

All three scripts resolve `config.yaml` and their input/output paths relative
to their own file location, so they can be run from any working directory.

## Running the pipeline for BiomedCLIP only

```bash
python extract_features.py --encoders biomedclip --tasks all --split train test
python train_probe.py --encoders biomedclip --tasks all
python plot_results.py
```
Note: `train_probe.py` overwrites `results/probe_results.csv` with whatever
encoders/tasks it was just run with. If you want one combined CSV across all
three encoders, run `train_probe.py` with `--encoders clip dino biomedclip`
together rather than one at a time.

## Reading the results

- `results/probe_results.csv`: one row per (encoder, task) — `accuracy`, `macro_f1`, `balanced_accuracy`, `best_epoch`, `encoder`, `task`.
- `results/plots/*.png`: visual comparison across encoders/tasks.
- W&B dashboard (`https://wandb.ai/<entity>/orena-probing`): per-epoch train/val loss curves for every run, grouped by task so you can compare encoders on the same task side by side.
- `probes/checkpoints/<encoder>/<task>.pt`: reload a trained probe later without re-running training:
  ```python
  import torch
  from probes.linear_probe import LinearProbe

  ckpt = torch.load("probes/checkpoints/biomedclip/grasped_by_instrument.pt", weights_only=False)
  probe = LinearProbe(ckpt["in_dim"], ckpt["num_classes"])
  probe.load_state_dict(ckpt["state_dict"])
  ```

## Adding a new encoder

1. Create `encoders/<name>.py` with a class `<Name>Encoder(BaseEncoder)` implementing:
   - `_load()` — build/download the pretrained model (+ its processor/transform), set `self.embed_dim`.
   - `preprocess(images: list[PIL.Image])` — turn PIL images into whatever `forward_features` expects.
   - `forward_features(batch)` — run the frozen model, return a `[batch, embed_dim]` tensor.

   Use `encoders/clip.py` as a template if it's HF `transformers`-based, or `encoders/biomedClip.py` if it's `open_clip`-based.
2. Register it in `encoders/__init__.py`: import the class and add `"<name>": <Name>Encoder` to `ENCODER_REGISTRY`.
3. Add an entry under `encoders:` in `config.yaml` with its `model_id` and a `device` (any free `cuda:N`).
4. Run `extract_features.py --encoders <name> --tasks all` then `train_probe.py --encoders <name> --tasks all`.

No other file needs to change — `data/`, `probes/`, and both CLIs are
encoder-agnostic; they only ever call `build_encoder(name, device)` and
`encoder.encode(images)`.

## Tasks on hold (pair / window)

Of the 14 tasks in `probing_export/`, only the 6 **single-frame** tasks are
wired up end-to-end (`fo_class_identification`, `fo_class_identification_multilabel`,
`quadrant_localization`, `closest_to_center`, `grasped_by_instrument`,
`object_count`).

The remaining 8 — 1 **pair** task (`instance_reidentification`, two images per
row) and 7 **window** tasks (a sampled sequence of N frames per row) — need a
decision on how to pool multiple per-row images into one embedding before
they can be extracted/trained. `data/preprocess.py`'s `prepare_task()` and
both CLIs already detect these tasks via `task_input_type()` and print a
"not yet implemented, skipping" message rather than erroring, so adding them
later won't require touching the single-frame code path.
