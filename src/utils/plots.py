"""Figure generation for the README.

RULE: every number plotted here is read from a file on disk -- run logs, results.json,
baselines.json, clean_metrics.json, or the cached labels/splits. Nothing is hand-typed
into this module. If a value is not recorded anywhere, the figure is not built.

House style: 150 dpi, ~800 px wide, no chartjunk, no 3D, colour used only where it
carries information.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["ADVE", "Email", "Form", "Letter", "Memo",
           "News", "Note", "Report", "Resume", "Scientific"]

DPI = 150
W = 800 / DPI          # ~800 px wide
INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#c1443c"
BLUE = "#3b6ea5"

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#e0e0e0", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- 1
def training_curves(metrics_csv: Path, out: Path) -> Path:
    rows = list(csv.DictReader(metrics_csv.open()))
    ep = [int(r["epoch"]) for r in rows]
    trl = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    f1 = [float(r["val_macro_f1"]) for r in rows]

    best_i = int(np.argmax(f1))
    best_ep, best_f1 = ep[best_i], f1[best_i]
    last_ep = ep[-1]

    fig, ax = plt.subplots(2, 1, figsize=(W, W * 0.85), sharex=True)

    ax[0].plot(ep, trl, color=BLUE, lw=1.5, label="train")
    ax[0].plot(ep, val, color=ACCENT, lw=1.5, label="validation")
    ax[0].set_ylabel("cross-entropy loss")
    ax[0].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax[0].grid(axis="y", alpha=0.5)

    ax[1].plot(ep, f1, color=ACCENT, lw=1.5)
    ax[1].scatter([best_ep], [best_f1], color=ACCENT, s=28, zorder=5)
    ax[1].set_ylabel("validation macro-F1")
    ax[1].set_xlabel("epoch")
    ax[1].grid(axis="y", alpha=0.5)

    for a in ax:
        a.axvline(best_ep, color=INK, ls="--", lw=0.9, alpha=0.7)
        a.axvline(last_ep, color=MUTED, ls=":", lw=0.9)
    ax[1].annotate(f"checkpoint selected\nepoch {best_ep}, macro-F1 {best_f1:.4f}",
                   xy=(best_ep, best_f1), xytext=(6, -30),
                   textcoords="offset points", fontsize=7.5, color=INK,
                   ha="left", va="top")
    # Label the stop on the LOWER panel: the upper one has the legend in that corner.
    ax[1].annotate(f"early stop, epoch {last_ep}", xy=(last_ep, min(f1)),
                   xytext=(-4, 4), textcoords="offset points",
                   fontsize=7, color=MUTED, ha="right", va="bottom")

    fig.suptitle("Training run — selection on validation macro-F1", y=0.98, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out)


# --------------------------------------------------------------------------- 2
def confusion_matrix(results_json: Path, out: Path) -> tuple[Path, dict]:
    res = json.loads(results_json.read_text())
    cm = np.array(res["confusion"])
    rown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(W, W * 0.92))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(10):
        for j in range(10):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if rown[i, j] > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("share of true class (row-normalised)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Confusion matrix — cells show raw counts, colour shows recall",
                 fontsize=9.5, pad=10)
    fig.tight_layout()

    # Where do Scientific's errors go? Reported, not assumed.
    si = CLASSES.index("Scientific")
    errs = {CLASSES[j]: int(cm[si, j]) for j in range(10) if j != si and cm[si, j] > 0}
    return _save(fig, out), {"scientific_errors": dict(sorted(
        errs.items(), key=lambda kv: -kv[1])), "scientific_recall": float(rown[si, si])}


# --------------------------------------------------------------------------- 3
def per_class_bars(results_json: Path, out: Path) -> Path:
    res = json.loads(results_json.read_text())
    pc = res["per_class"]
    order = sorted(CLASSES, key=lambda c: pc[c]["f1"])      # ascending: worst at top
    y = np.arange(len(order))
    h = 0.26

    fig, ax = plt.subplots(figsize=(W, W * 0.72))
    ax.barh(y + h, [pc[c]["precision"] for c in order], h, label="precision", color=BLUE)
    ax.barh(y, [pc[c]["recall"] for c in order], h, label="recall", color=ACCENT)
    ax.barh(y - h, [pc[c]["f1"] for c in order], h, label="F1", color=INK)

    ax.set_yticks(y, [f"{c}  (n={pc[c]['support']})" for c in order], fontsize=7.5)
    ax.invert_yaxis()          # worst F1 at the TOP, so the story reads downward
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("score")
    ax.grid(axis="x", alpha=0.5)
    # Outside the axes: every in-plot corner is occupied by a bar at this sort order.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=7.5)
    ax.set_title("Per-class performance on the test split, worst F1 first",
                 fontsize=9.5, pad=8)
    for i, c in enumerate(order):
        ax.text(pc[c]["f1"] + 0.012, i - h, f"{pc[c]['f1']:.3f}",
                va="center", fontsize=6.8, color=INK)
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- 4
def class_distribution(labels_npy: Path, splits_json: Path, out: Path) -> Path:
    lab = np.load(labels_npy)
    sp = {k: np.array(v) for k, v in json.loads(splits_json.read_text()).items()}
    counts = {s: np.bincount(lab[idx], minlength=10) for s, idx in sp.items()}
    total = counts["train"] + counts["val"] + counts["test"]
    order = np.argsort(-total)

    fig, ax = plt.subplots(figsize=(W, W * 0.62))
    y = np.arange(10)
    left = np.zeros(10)
    for split, colour in (("train", BLUE), ("val", "#7fa8cd"), ("test", ACCENT)):
        v = counts[split][order]
        ax.barh(y, v, left=left, color=colour, label=f"{split} ({counts[split].sum()})")
        left += v
    ax.set_yticks(y, [CLASSES[i] for i in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("images")
    ax.grid(axis="x", alpha=0.5)
    ax.legend(loc="lower right", fontsize=7.5)
    for i, idx in enumerate(order):
        ax.text(total[idx] + 12, i, str(total[idx]), va="center",
                fontsize=6.8, color=MUTED)
    imb = total.max() / total.min()
    ax.set_title(f"Class distribution across the stratified split "
                 f"({imb:.2f}× imbalance)", fontsize=9.5, pad=8)
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------- 5
def baseline_comparison(baselines_json: Path, clean_json: Path, out: Path,
                        ceiling: float | None = None) -> Path:
    """Two stacked panels rather than one crowded axis.

    The earlier single-panel version put the step annotations, the ceiling label and
    the bar values all in the same vertical band, where they collided with each other
    and with the bars. Splitting accuracy from the step-size commentary removes the
    contention entirely.
    """
    base = {b["name"]: b for b in json.loads(baselines_json.read_text())}
    clean = json.loads(clean_json.read_text())["clean"]

    names = ["majority\nclass", "logreg\n8×8", "logreg\n32×32", "DocCNN\n(ours)"]
    acc = [base["majority class"]["acc"], base["logreg 8x8"]["acc"],
           base["logreg 32x32"]["acc"], clean["acc"]]
    f1 = [base["majority class"]["macro_f1"], base["logreg 8x8"]["macro_f1"],
          base["logreg 32x32"]["macro_f1"], clean["macro_f1"]]

    x = np.arange(4)
    w = 0.36
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W, W * 0.78), sharex=True,
                                 gridspec_kw={"height_ratios": [3.1, 1]})

    ax.bar(x - w / 2, acc, w, label="accuracy", color=[MUTED, MUTED, MUTED, BLUE])
    ax.bar(x + w / 2, f1, w, label="macro-F1", color=["#c9c9c9"] * 3 + [ACCENT])
    for i in range(4):
        ax.text(x[i] - w / 2, acc[i] + 0.02, f"{acc[i]*100:.1f}%", ha="center",
                fontsize=7, color=INK)
        ax.text(x[i] + w / 2, f1[i] + 0.02, f"{f1[i]:.3f}", ha="center",
                fontsize=7, color=INK)

    if ceiling is not None:
        ax.axhline(ceiling, ls="--", lw=1, color=INK, alpha=0.55)
        ax.text(3.55, ceiling + 0.018,
                f"~{ceiling*100:.0f}% label-noise ceiling\n(arXiv:2412.13140)",
                ha="right", va="bottom", fontsize=6.6, color=INK, alpha=0.85,
                linespacing=1.3)

    ax.set_ylim(0, 1.24)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.5)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, bbox_to_anchor=(0.0, 1.0))
    ax.set_title("Against the floors — test split", fontsize=9.5, pad=8)

    # Lower panel: the step from each model to the next, which is the actual point.
    steps = [acc[i + 1] - acc[i] for i in range(3)]
    bx.bar([0.5, 1.5, 2.5], [s * 100 for s in steps], 0.5,
           color=[MUTED, MUTED, INK])
    for i, s in enumerate(steps):
        bx.text(0.5 + i, s * 100 + 1.2, f"+{s*100:.2f}", ha="center",
                fontsize=7, color=INK)
    bx.text(1.5, 11, "16× the pixel features,\nalmost no gain", ha="center",
            va="bottom", fontsize=6.6, color=MUTED, linespacing=1.3)
    bx.set_ylim(0, 56)
    bx.set_ylabel("step\n(acc. pts)", fontsize=7.5)
    bx.set_xticks(x, names, fontsize=7.5)
    bx.set_xlim(-0.6, 3.6)
    bx.grid(axis="y", alpha=0.5)

    fig.tight_layout()
    return _save(fig, out)
