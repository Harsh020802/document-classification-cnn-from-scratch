"""Phase 1 step 8 -- error analysis.

Three outputs:
  1. Confusion matrix, so which classes get confused is visible rather than inferred
  2. Most-CONFIDENT errors -- the model was sure and wrong. Given the dataset's known
     11.7% label-error rate, some of these are expected to be the MODEL being right and
     the LABEL being wrong. Finding those is the point (bentrevett's approach, D-003).
  3. Per-class metrics against class size, to see whether small classes underperform

Usage:
  PYTHONPATH=. python scripts/10_error_analysis.py                # newest run
  PYTHONPATH=. python scripts/10_error_analysis.py --run <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import ConfusionMatrixDisplay

from src.data.splits import load_splits

CLASSES = ["ADVE", "Email", "Form", "Letter", "Memo",
           "News", "Note", "Report", "Resume", "Scientific"]


def latest_run() -> Path:
    runs = sorted((REPO / "outputs" / "runs").glob("*/results.json"))
    if not runs:
        sys.exit("no completed run found (looked for outputs/runs/*/results.json)")
    return runs[-1].parent


def plot_confusion(cm: np.ndarray, out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        values_format="d", cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title("counts")
    # Row-normalised: with a 5.2x imbalance, raw counts make big classes look good
    # simply for being big. Row-normalised shows RECALL per class.
    rn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ConfusionMatrixDisplay(rn, display_labels=CLASSES).plot(
        values_format=".2f", cmap="Blues", ax=axes[1], colorbar=False)
    axes[1].set_title("row-normalised (recall per true class)")
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


def plot_confident_errors(cache, test_idx, y_true, y_pred, probs,
                          out: Path, n: int = 12) -> list[dict]:
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("  no errors to plot")
        return []
    conf = probs[wrong, y_pred[wrong]]
    order = wrong[np.argsort(-conf)][:n]

    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 5.2 * rows))
    recs = []
    for ax, k in zip(axes.flat, order):
        img = np.asarray(cache[test_idx[k]])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        p_pred = probs[k, y_pred[k]]
        p_true = probs[k, y_true[k]]
        ax.set_title(f"pred {CLASSES[y_pred[k]]} ({p_pred:.2f})\n"
                     f"true {CLASSES[y_true[k]]} ({p_true:.2f})",
                     fontsize=10, color="crimson")
        ax.axis("off")
        recs.append({"test_pos": int(k), "dataset_idx": int(test_idx[k]),
                     "pred": CLASSES[y_pred[k]], "true": CLASSES[y_true[k]],
                     "p_pred": float(p_pred), "p_true": float(p_true)})
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    fig.suptitle("Most-confident errors -- some of these are expected to be LABEL "
                 "errors, not model errors (11.7% known rate)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=95, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = Path(args.run) if args.run else latest_run()
    print(f"run: {run.relative_to(REPO) if run.is_relative_to(REPO) else run}")

    res = json.loads((run / "results.json").read_text())
    preds = np.load(run / "test_predictions.npy", allow_pickle=True).item()
    y_true, y_pred, probs = preds["y_true"], preds["y_pred"], preds["probs"]
    cm = np.array(res["confusion"])

    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    cache = np.load(REPO / cfg["data"]["cache_dir"] / "images.npy", mmap_mode="r")
    test_idx = load_splits(REPO / cfg["data"]["splits_dir"] / "splits.json")["test"]

    print(f"\nTEST  acc {res['test_acc']:.4f}   macro-F1 {res['test_macro_f1']:.4f}")

    figs = REPO / "outputs" / "figures"
    plot_confusion(cm, figs / "04_confusion_matrix.png",
                   f"{res['head']} seed {res['seed']} -- test acc {res['test_acc']:.3f}, "
                   f"macro-F1 {res['test_macro_f1']:.3f}")
    recs = plot_confident_errors(cache, test_idx, y_true, y_pred, probs,
                                 figs / "05_confident_errors.png")

    # Which pairs get confused most, off-diagonal.
    print("\n  most-confused class PAIRS (off-diagonal, both directions):")
    pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], CLASSES[i], CLASSES[j]))
    for n, a, b in sorted(pairs, reverse=True)[:8]:
        print(f"    {n:>3}x  true {a:<12} -> predicted {b}")

    print(f"\n  {'class':<12}{'n':>5}{'prec':>8}{'recall':>8}{'F1':>8}   vs class size")
    print("  " + "-" * 58)
    for c in CLASSES:
        r = res["per_class"][c]
        bar = "#" * int(r["f1"] * 30)
        print(f"  {c:<12}{r['support']:>5}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}   {bar}")

    sizes = np.array([res["per_class"][c]["support"] for c in CLASSES])
    f1s = np.array([res["per_class"][c]["f1"] for c in CLASSES])
    corr = float(np.corrcoef(sizes, f1s)[0, 1])
    print(f"\n  correlation(class size, F1) = {corr:+.3f}")
    print("  A strong positive value means small classes are being neglected despite")
    print("  the inverse-frequency loss weighting.")

    (run / "error_analysis.json").write_text(json.dumps(
        {"confident_errors": recs, "size_f1_corr": corr}, indent=2))
    print(f"\n  saved -> {(run / 'error_analysis.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
