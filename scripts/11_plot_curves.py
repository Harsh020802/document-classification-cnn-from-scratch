"""Plot training curves for a run (default: the newest)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.viz import plot_curves

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=None)
a = ap.parse_args()

run = Path(a.run) if a.run else sorted((REPO / "outputs" / "runs").glob("*/metrics.csv"))[-1].parent
res_path = run / "results.json"
title = run.name
if res_path.exists():
    r = json.loads(res_path.read_text())
    title = (f"{r['head']} seed {r['seed']} -- test acc {r['test_acc']:.3f}, "
             f"macro-F1 {r['test_macro_f1']:.3f}")
out = plot_curves(run / "metrics.csv", REPO / "outputs" / "figures" / "06_training_curves.png", title)
print(f"wrote {out.relative_to(REPO)}")
