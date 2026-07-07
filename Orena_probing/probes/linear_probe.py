from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from torch import nn


class LinearProbe(nn.Module):
    """embedding -> linear layer -> class prediction. Only this layer is trained."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def _to_tensor(x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, multi_label: bool) -> dict:
    if multi_label:
        # Per-label binary metrics, macro-averaged across classes.
        n_classes = y_true.shape[1]
        per_label_balanced_acc = [
            balanced_accuracy_score(y_true[:, c], y_pred[:, c])
            for c in range(n_classes)
            if len(np.unique(y_true[:, c])) > 1
        ]
        return {
            "accuracy": accuracy_score(y_true.ravel(), y_pred.ravel()),  # element-wise (Hamming) accuracy
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "balanced_accuracy": float(np.mean(per_label_balanced_acc)) if per_label_balanced_acc else float("nan"),
        }
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def train_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    multi_label: bool,
    num_classes: int,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 10,
    device: str = "cuda",
    seed: int = 42,
    on_epoch: Callable[[int, dict], None] | None = None,
) -> tuple[dict, LinearProbe]:
    """Train a linear classifier on frozen embeddings.

    Model selection (early stopping, best-checkpoint) is driven entirely by the
    validation split. The test split is only ever touched once, after training
    is finished, to compute the final reported metrics — it never influences
    which epoch gets kept.

    `on_epoch(epoch, {"train_loss": ..., "val_loss": ...})` is called after every
    epoch if given — used to stream per-epoch curves to an external logger (e.g.
    wandb) without coupling this module to any particular logging backend.

    Returns (metrics, model) — model holds the best-checkpoint weights (by val loss).
    """
    torch.manual_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"

    in_dim = X_train.shape[1]
    model = LinearProbe(in_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    label_dtype = torch.float32 if multi_label else torch.long

    X_train_t = _to_tensor(X_train).to(device)
    y_train_t = _to_tensor(y_train, dtype=label_dtype).to(device)
    X_val_t = _to_tensor(X_val).to(device)
    y_val_t = _to_tensor(y_val, dtype=label_dtype).to(device)

    n = X_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        train_loss_sum = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            logits = model(X_train_t[idx])
            loss = loss_fn(logits, y_train_t[idx])
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(idx)
        train_loss = train_loss_sum / n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if on_epoch is not None:
            on_epoch(epoch, {"train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_test_t = _to_tensor(X_test).to(device)
        logits = model(X_test_t).cpu().numpy()

    if multi_label:
        y_pred = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    else:
        y_pred = logits.argmax(axis=1)

    metrics = compute_metrics(y_test, y_pred, multi_label)
    metrics["best_epoch"] = best_epoch
    return metrics, model
