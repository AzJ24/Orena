"""Visualize results/probe_results.csv: grouped bar charts comparing encoders
per task, one chart per metric, plus a macro-F1 heatmap.

Usage:
    python plot_results.py
    python plot_results.py --results results/probe_results.csv --out-dir results/plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["accuracy", "macro_f1", "balanced_accuracy"]
BASE_DIR = Path(__file__).resolve().parent

# Vision-tower parameter counts (millions), measured by counting the params of
# each encoder's visual tower only (text/LLM towers excluded). Shown in the
# legend so model capacity is visible alongside performance.
VISION_PARAMS_M = {
    "dinov3": 85.7,
    "levljepa": 85.8,
    "biomedclip": 86.2,
    "dino": 86.6,
    "clip": 87.8,
    "gemma": 167.4,
    "levjepa": 305.5,
    "qwen": 333.5,
}


def _legend_label(encoder: str) -> str:
    """Encoder name + its vision-tower param count, e.g. 'dino (87M)'."""
    p = VISION_PARAMS_M.get(encoder)
    return f"{encoder} ({p:.0f}M)" if p is not None else encoder


def plot_metric_bars(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    tasks = sorted(df["task"].unique())
    # Order encoders by vision-tower size (smallest -> largest) so the bars and
    # the legend read in a consistent, meaningful order.
    encoders = sorted(df["encoder"].unique(), key=lambda e: VISION_PARAMS_M.get(e, float("inf")))
    pivot = df.pivot(index="task", columns="encoder", values=metric).reindex(tasks)

    x = np.arange(len(tasks))
    width = 0.8 / len(encoders)
    # Center the whole group of bars on each task tick.
    offsets = (np.arange(len(encoders)) - (len(encoders) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(max(9, len(tasks) * 1.6), 5.5))
    for off, encoder in zip(offsets, encoders):
        ax.bar(x + off, pivot[encoder].values, width, label=_legend_label(encoder))

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Reserve a top band for the title + legend; bars fill the rest, centered.
    fig.subplots_adjust(top=0.80, bottom=0.24, left=0.07, right=0.98)

    # Legend as a clean horizontal strip on top, outside the axes so it never
    # covers the bars. Wrap to at most 4 columns per row.
    ncol = min(len(encoders), 4)
    ax.legend(
        title="Vision encoder (param count)",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=ncol,
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    # Title sits just above the legend strip (no empty gap).
    fig.suptitle(f"{metric.replace('_', ' ')} by encoder and task", y=0.985, fontweight="bold")

    out_path = out_dir / f"{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] saved {out_path}")


def plot_heatmap(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    tasks = sorted(df["task"].unique())
    encoders = sorted(df["encoder"].unique())
    pivot = df.pivot(index="encoder", columns="task", values=metric).reindex(index=encoders, columns=tasks)

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.2), 1 + len(encoders) * 0.8))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(len(encoders)))
    ax.set_yticklabels(encoders)
    ax.set_title(f"{metric} heatmap")
    for i in range(len(encoders)):
        for j in range(len(tasks)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    out_path = out_dir / f"{metric}_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_results] saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(BASE_DIR / "results" / "probe_results.csv"))
    parser.add_argument("--out-dir", default=str(BASE_DIR / "results" / "plots"))
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        plot_metric_bars(df, metric, out_dir)
    plot_heatmap(df, "macro_f1", out_dir)


if __name__ == "__main__":
    main()
