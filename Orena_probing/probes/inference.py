"""Inference-side helpers shared by the interactive notebook (inference_explorer.ipynb).

Two code paths, deliberately kept separate:
  - single image: runs the real frozen encoder + probe live, so embeddings and
    timings are genuine (not read from the features/ cache).
  - full test set: reuses the cached features/<encoder>/<task>/test.npz +
    probes/checkpoints/<encoder>/<task>.pt — instant, no encoder needed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from data.dataset import get_label_info, load_task
from encoders import BaseEncoder, build_encoder
from probes.linear_probe import LinearProbe, compute_metrics

BASE_DIR = Path(__file__).resolve().parent.parent

# Process-lifetime caches so repeated widget interactions don't reload models/datasets.
_encoder_cache: dict[str, BaseEncoder] = {}
_dataset_cache: dict[tuple[str, str], object] = {}


def load_config(path: str | None = None) -> dict:
    path = path or str(BASE_DIR / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def available_encoders_for_task(task: str, cfg: dict) -> list[str]:
    """Encoders that have both a trained checkpoint and cached test features for `task`."""
    checkpoints_dir = BASE_DIR / cfg["paths"]["checkpoints_dir"]
    features_dir = BASE_DIR / cfg["paths"]["features_dir"]
    return [
        name
        for name in cfg["encoders"]
        if (checkpoints_dir / name / f"{task}.pt").exists() and (features_dir / name / task / "test.npz").exists()
    ]


def get_encoder(encoder_name: str, cfg: dict, device: str | None = None) -> BaseEncoder:
    """Build (or reuse) the frozen encoder for `encoder_name`."""
    if encoder_name not in _encoder_cache:
        device = device or cfg["encoders"][encoder_name].get("device", cfg["training"]["device"])
        _encoder_cache[encoder_name] = build_encoder(encoder_name, device=device)
    return _encoder_cache[encoder_name]


def get_test_dataset(task: str, cfg: dict):
    """Load (and cache) the test split of a task, for picking a single image by qID."""
    key = (task, "test")
    if key not in _dataset_cache:
        _dataset_cache[key] = load_task(
            task,
            "test",
            source=cfg["data"]["source"],
            local_dir=str(BASE_DIR / cfg["data"]["local_dir"]),
            hub_repo=cfg["data"]["hub_repo"],
        )
    return _dataset_cache[key]


def load_probe(encoder_name: str, task: str, cfg: dict, device: str) -> tuple[LinearProbe, dict]:
    """Load a trained probe checkpoint, return (model, checkpoint_dict)."""
    ckpt_path = BASE_DIR / cfg["paths"]["checkpoints_dir"] / encoder_name / f"{task}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} — run train_probe.py --encoders {encoder_name} --tasks {task} first."
        )
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = LinearProbe(ckpt["in_dim"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@dataclass
class SingleImageResult:
    image: Image.Image
    embedding: np.ndarray          # [embed_dim]
    probs: np.ndarray              # [num_classes] (softmax for single-label, sigmoid for multi-label)
    pred_label: object             # class name (single-label) or list of class names (multi-label)
    true_label: object
    multi_label: bool
    class_names: list[str]
    encode_ms: float
    probe_ms: float
    total_ms: float


def run_single_image(
    encoder_name: str, task: str, qid: str, cfg: dict, warmup: bool = True
) -> SingleImageResult:
    """Run one test-set image through the real encoder + trained probe, with timing.

    If `warmup` is True, one throwaway forward pass runs first so the reported
    times reflect steady-state inference, not the one-off CUDA init / cuDNN
    autotuning cost paid on the very first call after an encoder is loaded.
    """
    ds = get_test_dataset(task, cfg)
    row = ds[ds["qID"].index(qid)]
    image = row["image"]
    label_info = get_label_info(ds)
    true_raw = row[label_info.label_key]
    true_label = (
        [label_info.class_names[i] for i in true_raw] if label_info.multi_label else label_info.class_names[true_raw]
    )

    encoder = get_encoder(encoder_name, cfg)
    model, ckpt = load_probe(encoder_name, task, cfg, device=encoder.device)

    def _sync():
        # CUDA kernels are async — block until the GPU is actually done so the
        # timers below measure real compute, not just kernel-launch overhead.
        if torch.cuda.is_available() and str(encoder.device).startswith("cuda"):
            torch.cuda.synchronize(encoder.device)

    if warmup:
        with torch.no_grad():
            warm = encoder.encode([image], batch_size=1)
            model(torch.as_tensor(warm, dtype=torch.float32).to(encoder.device))
        _sync()

    # End-to-end timer: starts the instant the image enters the pipeline
    # (preprocessing + encoder), runs through the probe, and stops only once the
    # predicted class label is fully resolved.
    _sync()
    t0 = time.perf_counter()

    embedding = encoder.encode([image], batch_size=1)  # preprocess + encoder forward -> [1, embed_dim]
    _sync()
    t1 = time.perf_counter()

    with torch.no_grad():
        x = torch.as_tensor(embedding, dtype=torch.float32).to(encoder.device)
        logits = model(x)
        if ckpt["multi_label"]:
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    if ckpt["multi_label"]:
        pred_label = [ckpt["class_names"][i] for i in np.where(probs > 0.5)[0]]
    else:
        pred_label = ckpt["class_names"][int(probs.argmax())]
    _sync()
    t2 = time.perf_counter()

    return SingleImageResult(
        image=image,
        embedding=embedding[0],
        probs=probs,
        pred_label=pred_label,
        true_label=true_label,
        multi_label=ckpt["multi_label"],
        class_names=list(ckpt["class_names"]),
        encode_ms=(t1 - t0) * 1000,
        probe_ms=(t2 - t1) * 1000,
        total_ms=(t2 - t0) * 1000,
    )


@dataclass
class TestSetResult:
    metrics: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    multi_label: bool
    class_names: list[str]


def run_test_set(encoder_name: str, task: str, cfg: dict) -> TestSetResult:
    """Evaluate the trained probe on the full cached test set."""
    features_dir = BASE_DIR / cfg["paths"]["features_dir"]
    npz = np.load(features_dir / encoder_name / task / "test.npz", allow_pickle=True)
    X_test, y_test = npz["embeddings"], npz["labels"]
    class_names = list(npz["class_names"])
    multi_label = bool(npz["multi_label"])

    device = cfg["encoders"][encoder_name].get("device", cfg["training"]["device"])
    device = device if torch.cuda.is_available() else "cpu"
    model, _ = load_probe(encoder_name, task, cfg, device=device)

    with torch.no_grad():
        x = torch.as_tensor(X_test, dtype=torch.float32).to(device)
        logits = model(x).cpu().numpy()

    if multi_label:
        y_pred = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    else:
        y_pred = logits.argmax(axis=1)

    metrics = compute_metrics(y_test, y_pred, multi_label)
    return TestSetResult(metrics=metrics, y_true=y_test, y_pred=y_pred, multi_label=multi_label, class_names=class_names)
