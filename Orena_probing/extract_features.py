"""Step 1 + 2 of the linear-probing methodology: freeze each encoder, run every
image through it, and cache the fixed embeddings to disk (one .npz per
encoder/task/split).

Usage:
    python extract_features.py --encoders clip dino biomedclip --tasks all --split train test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from data.dataset import SINGLE_FRAME_TASKS, get_label_info, load_task, task_input_type
from data.preprocess import prepare_task
from encoders import build_encoder


BASE_DIR = Path(__file__).resolve().parent


def load_config(path: str | None = None) -> dict:
    path = path or str(BASE_DIR / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def extract_one(encoder, task: str, split: str, cfg: dict, out_dir: Path) -> None:
    ds = load_task(
        task,
        split,
        source=cfg["data"]["source"],
        local_dir=str(BASE_DIR / cfg["data"]["local_dir"]),
        hub_repo=cfg["data"]["hub_repo"],
    )
    label_info = get_label_info(ds)
    prepared = prepare_task(task, ds, label_info)
    if prepared is None:
        return
    images, labels = prepared

    # Per-encoder batch_size override (config.yaml encoders.<name>.batch_size),
    # for encoders whose forward is too memory-heavy at the global batch size.
    batch_size = cfg["encoders"][encoder.name].get("batch_size", cfg["training"]["batch_size"])
    print(f"[extract] {encoder.name} / {task} / {split}: {len(images)} images (batch_size={batch_size})")
    embeddings = encoder.encode(images, batch_size=batch_size)

    task_dir = out_dir / encoder.name / task
    task_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        task_dir / f"{split}.npz",
        embeddings=embeddings,
        labels=labels,
        qids=np.asarray(ds["qID"]),
        multi_label=label_info.multi_label,
        num_classes=label_info.num_classes,
        class_names=np.asarray(label_info.class_names),
    )


def main() -> None:
    cfg = load_config()
    all_tasks = SINGLE_FRAME_TASKS + cfg["tasks"]["pair"] + cfg["tasks"]["window"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", default=list(cfg["encoders"].keys()))
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--split", nargs="+", default=["train", "test"])
    parser.add_argument(
        "--device",
        default=None,
        help="Override device for every encoder. Omit to use each encoder's "
        "own 'device' from config.yaml (e.g. clip on cuda:0, dino on cuda:1) "
        "so different models run on different GPUs.",
    )
    args = parser.parse_args()

    tasks = all_tasks if args.tasks == ["all"] else args.tasks
    out_dir = BASE_DIR / cfg["paths"]["features_dir"]

    for encoder_name in args.encoders:
        device = args.device or cfg["encoders"][encoder_name].get("device", cfg["training"]["device"])
        encoder = build_encoder(encoder_name, device=device)
        for task in tasks:
            if task_input_type(task) != "single_frame":
                print(f"[extract] task '{task}' is pair/window, not yet implemented — skipping.")
                continue
            for split in args.split:
                extract_one(encoder, task, split, cfg, out_dir)


if __name__ == "__main__":
    main()
