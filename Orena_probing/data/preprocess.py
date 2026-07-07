from __future__ import annotations

import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split

from .dataset import LabelInfo, task_input_type


def get_single_frame_images(ds: Dataset) -> list:
    """One PIL image per row, for tasks in SINGLE_FRAME_TASKS."""
    return list(ds["image"])


def get_labels(ds: Dataset, label_info: LabelInfo) -> np.ndarray:
    """Row labels as either int class ids (single-label) or a [N, C] multi-hot matrix."""
    raw = ds[label_info.label_key]
    if not label_info.multi_label:
        return np.asarray(raw, dtype=np.int64)
    multi_hot = np.zeros((len(raw), label_info.num_classes), dtype=np.float32)
    for i, row_labels in enumerate(raw):
        multi_hot[i, row_labels] = 1.0
    return multi_hot


def prepare_task(task: str, ds: Dataset, label_info: LabelInfo):
    """Return (images, labels) ready for an encoder, or None for tasks on hold."""
    input_type = task_input_type(task)
    if input_type != "single_frame":
        print(
            f"[preprocess] task '{task}' has input_type='{input_type}' "
            "(pair/window) — not yet implemented, skipping."
        )
        return None
    images = get_single_frame_images(ds)
    labels = get_labels(ds, label_info)
    return images, labels


def split_train_val(
    X: np.ndarray, y: np.ndarray, multi_label: bool, val_split: float = 0.2, seed: int = 42
):
    """Carve a validation set out of the train split, for early stopping.

    Stratifies on the label so each class keeps roughly the same proportion in
    both halves (balanced). Falls back to a plain random split if stratifying
    isn't possible — e.g. a single-label class with too few rows to appear in
    both halves, or multi-label targets (no single column to stratify on).

    Returns (X_train, X_val, y_train, y_val).
    """
    stratify = y if (not multi_label and np.min(np.bincount(y)) >= 2) else None
    try:
        return train_test_split(X, y, test_size=val_split, random_state=seed, stratify=stratify)
    except ValueError:
        return train_test_split(X, y, test_size=val_split, random_state=seed, stratify=None)
