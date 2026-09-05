"""Plotting helpers: training curves and comparison tables.

matplotlib rather than TensorBoard (D-003): CSV logs are diffable across ablation arms,
and it is one less dependency on a disk-constrained machine.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(csv_path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols


def plot_curves(csv_path: Path, out: Path, title: str = "") -> None:
    """Train/val loss, accuracy, macro-F1, and the LR schedule.

    The train-vs-val GAP is the thing to read here, not the absolute values: with ~2,438
    training images and no pretrained weights, overfitting is the dominant risk, and the
    widening gap between the two curves is how you see it happening.
    """
    m = read_metrics(csv_path)
    ep = m["epoch"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    ax[0, 0].plot(ep, m["train_loss"], label="train", lw=1.8)
    ax[0, 0].plot(ep, m["val_loss"], label="val", lw=1.8)
    ax[0, 0].set_title("loss"); ax[0, 0].set_xlabel("epoch"); ax[0, 0].legend()

    ax[0, 1].plot(ep, m["train_acc"], label="train", lw=1.8)
    ax[0, 1].plot(ep, m["val_acc"], label="val", lw=1.8)
    ax[0, 1].set_title("accuracy"); ax[0, 1].set_xlabel("epoch"); ax[0, 1].legend()

    ax[1, 0].plot(ep, m["train_macro_f1"], label="train", lw=1.8)
    ax[1, 0].plot(ep, m["val_macro_f1"], label="val", lw=1.8)
    best = max(m["val_macro_f1"])
    bi = m["val_macro_f1"].index(best)
    ax[1, 0].axvline(ep[bi], color="grey", ls=":", lw=1)
    ax[1, 0].scatter([ep[bi]], [best], color="crimson", zorder=5, s=35)
    ax[1, 0].annotate(f"best {best:.4f} @ ep{int(ep[bi])}",
                      (ep[bi], best), textcoords="offset points",
                      xytext=(6, -14), fontsize=9, color="crimson")
    ax[1, 0].set_title("macro-F1 (the selection metric)")
    ax[1, 0].set_xlabel("epoch"); ax[1, 0].legend()

    gap = [t - v for t, v in zip(m["train_acc"], m["val_acc"])]
    ax[1, 1].plot(ep, gap, color="darkorange", lw=1.8)
    ax[1, 1].axhline(0, color="grey", lw=0.8)
    ax[1, 1].set_title("train acc - val acc  (the overfitting gauge)")
    ax[1, 1].set_xlabel("epoch")

    for a in ax.flat:
        a.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out
