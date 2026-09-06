"""Regenerate every README figure in one command.

  ./run.sh scripts/make_figures.py

Every number comes from a file on disk. If a required file is missing the script says
which, and skips that figure rather than inventing values.

The ~88% label-noise ceiling drawn on figure 5 is NOT a measurement of ours -- it comes
from arXiv:2412.13140, which found 11.7% of Tobacco-3482 labels wrong. It is declared here
explicitly as an external citation rather than buried in the plotting code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.plots import (ablation, baseline_comparison, class_distribution,
                             confusion_matrix, per_class_bars, training_curves)

RUN = REPO / "outputs" / "runs" / "0906_002054_gap3_seed0_first"
FIGS = REPO / "outputs" / "figures"

# External, cited, not measured here. arXiv:2412.13140 -- 11.7% label error rate.
LABEL_NOISE_CEILING = 0.88


def need(p: Path, what: str) -> bool:
    if p.exists():
        return True
    print(f"  SKIPPED {what}: missing {p.relative_to(REPO)}")
    return False


def main() -> None:
    made = []
    print("regenerating figures from files on disk\n")

    if need(RUN / "metrics.csv", "training curves"):
        p = training_curves(RUN / "metrics.csv", FIGS / "fig1_training_curves.png")
        made.append(p)
        print(f"  wrote {p.relative_to(REPO)}")

    if need(RUN / "results.json", "confusion matrix"):
        p, info = confusion_matrix(RUN / "results.json",
                                   FIGS / "fig2_confusion_matrix.png")
        made.append(p)
        print(f"  wrote {p.relative_to(REPO)}")
        print(f"    Scientific recall {info['scientific_recall']:.3f}; "
              f"its errors go to: " +
              ", ".join(f"{k} {v}" for k, v in info["scientific_errors"].items()))

        p = per_class_bars(RUN / "results.json", FIGS / "fig3_per_class.png")
        made.append(p)
        print(f"  wrote {p.relative_to(REPO)}")

    if need(REPO / "data" / "cache" / "labels.npy", "class distribution"):
        p = class_distribution(REPO / "data" / "cache" / "labels.npy",
                               REPO / "data" / "splits" / "splits.json",
                               FIGS / "fig4_class_distribution.png")
        made.append(p)
        print(f"  wrote {p.relative_to(REPO)}")

    cm = REPO / "outputs" / "clean_metrics.json"
    if need(cm, "baseline comparison") and need(REPO / "outputs" / "baselines.json",
                                                "baseline comparison"):
        p = baseline_comparison(REPO / "outputs" / "baselines.json", cm,
                                FIGS / "fig5_baselines.png",
                                ceiling=LABEL_NOISE_CEILING)
        made.append(p)
        print(f"  wrote {p.relative_to(REPO)}")
        c = json.loads(cm.read_text())
        print(f"    CNN plotted at {c['clean']['acc']:.4f} / "
              f"{c['clean']['macro_f1']:.4f} (leakage-corrected, n={c['n_clean']})")

    ab = REPO / "outputs" / "ablation_results.json"
    fl = REPO / "outputs" / "seed_floor.json"
    if need(ab, "ablation") and need(fl, "ablation"):
        p = ablation(ab, fl, FIGS / "fig6_ablation.png")
        made.append(p)
        print(f"  wrote {p.relative_to(REPO)}")

    print(f"\n  {len(made)} figures written to outputs/figures/")


if __name__ == "__main__":
    main()
