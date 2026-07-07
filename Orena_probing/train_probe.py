"""Step 3 + 4 of the linear-probing methodology: train a linear classifier on
top of cached frozen embeddings and report accuracy / macro-F1 / balanced
accuracy. Run extract_features.py first to populate features/.

Usage:
    python train_probe.py --encoders clip dino biomedclip --tasks all
    python train_probe.py --encoders clip --tasks fo_class_identification
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
import yaml

from data.dataset import SINGLE_FRAME_TASKS, task_input_type
from data.preprocess import split_train_val
from probes.linear_probe import train_linear_probe


BASE_DIR = Path(__file__).resolve().parent


def load_config(path: str | None = None) -> dict:
    path = path or str(BASE_DIR / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_features(features_dir: Path, encoder: str, task: str, split: str):
    npz = np.load(features_dir / encoder / task / f"{split}.npz", allow_pickle=True)
    return (
        npz["embeddings"],
        npz["labels"],
        bool(npz["multi_label"]),
        int(npz["num_classes"]),
        list(npz["class_names"]),
    )


def run_one(
    encoder: str,
    task: str,
    features_dir: Path,
    training_cfg: dict,
    device: str,
    checkpoints_dir: Path,
    wandb_cfg: dict,
) -> dict | None:
    train_path = features_dir / encoder / task / "train.npz"
    test_path = features_dir / encoder / task / "test.npz"
    if not train_path.exists() or not test_path.exists():
        print(f"[train_probe] missing features for {encoder}/{task}, run extract_features.py first — skipping.")
        return None

    X_train_full, y_train_full, multi_label, num_classes, class_names = load_features(
        features_dir, encoder, task, "train"
    )
    X_test, y_test, _, _, _ = load_features(features_dir, encoder, task, "test")

    X_train, X_val, y_train, y_val = split_train_val(
        X_train_full,
        y_train_full,
        multi_label=multi_label,
        val_split=training_cfg["val_split"],
        seed=training_cfg["seed"],
    )

    print(
        f"[train_probe] {encoder} / {task}: "
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)} multi_label={multi_label}"
    )

    run = wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg.get("entity"),
        mode=wandb_cfg["mode"],
        name=f"{encoder}/{task}",
        group=task,       # group by task in the dashboard, to compare encoders on the same task
        job_type=encoder,
        reinit="create_new",
        config={
            "encoder": encoder,
            "task": task,
            "multi_label": multi_label,
            "num_classes": num_classes,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            **{k: v for k, v in training_cfg.items() if k != "device"},
            "device": device,
        },
    )

    def on_epoch(epoch: int, logs: dict) -> None:
        run.log({"epoch": epoch, **logs})

    metrics, model = train_linear_probe(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        multi_label=multi_label,
        num_classes=num_classes,
        epochs=training_cfg["epochs"],
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
        batch_size=training_cfg["batch_size"],
        patience=training_cfg["patience"],
        device=device,
        seed=training_cfg["seed"],
        on_epoch=on_epoch,
    )
    metrics.update(encoder=encoder, task=task)

    run.log({f"test/{k}": v for k, v in metrics.items() if k not in ("encoder", "task")})
    run.finish()

    ckpt_dir = checkpoints_dir / encoder
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{task}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": X_train_full.shape[1],
            "num_classes": num_classes,
            "multi_label": multi_label,
            "class_names": class_names,
            "encoder": encoder,
            "task": task,
            "metrics": metrics,
        },
        ckpt_path,
    )
    print(f"[train_probe] saved checkpoint to {ckpt_path}")
    return metrics


def main() -> None:
    cfg = load_config()
    all_tasks = SINGLE_FRAME_TASKS + cfg["tasks"]["pair"] + cfg["tasks"]["window"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", default=list(cfg["encoders"].keys()))
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument(
        "--device",
        default=None,
        help="Override device for every encoder's probe training. Omit to use "
        "each encoder's own 'device' from config.yaml.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Override wandb.mode from config.yaml (e.g. --wandb-mode disabled to turn off logging).",
    )
    args = parser.parse_args()

    training_cfg = dict(cfg["training"])
    wandb_cfg = dict(cfg["wandb"])
    if args.wandb_mode is not None:
        wandb_cfg["mode"] = args.wandb_mode

    tasks = all_tasks if args.tasks == ["all"] else args.tasks
    features_dir = BASE_DIR / cfg["paths"]["features_dir"]
    results_dir = BASE_DIR / cfg["paths"]["results_dir"]
    checkpoints_dir = BASE_DIR / cfg["paths"]["checkpoints_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        if task_input_type(task) != "single_frame":
            print(f"[train_probe] task '{task}' is pair/window, not yet implemented — skipping.")
            continue
        for encoder in args.encoders:
            device = args.device or cfg["encoders"][encoder].get("device", cfg["training"]["device"])
            metrics = run_one(encoder, task, features_dir, training_cfg, device, checkpoints_dir, wandb_cfg)
            if metrics is not None:
                results.append(metrics)

    if not results:
        print("[train_probe] no results produced.")
        return

    df = pd.DataFrame(results)
    out_path = results_dir / "probe_results.csv"

    # Merge into any existing CSV instead of overwriting it, so running one
    # encoder doesn't wipe results from earlier runs of other encoders. Rows are
    # keyed by (encoder, task) — a re-run of the same pair replaces its old row.
    if out_path.exists():
        prev = pd.read_csv(out_path)
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=["encoder", "task"], keep="last").reset_index(drop=True)

    df = df.sort_values(["task", "encoder"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
