"""Persist the leakage-corrected test metrics as a machine-readable file.

The 84.84% / 0.8282 figures existed only as prose in notes/RESULTS.md and D-043. Any
figure that needs them would have to hard-code them, which breaks the rule that every
plotted number comes from a file. This recomputes them from the saved predictions and the
identified contaminated indices, and writes outputs/clean_metrics.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import imagehash
import numpy as np
from PIL import Image

from src.data.splits import load_splits
from src.engine.metrics import macro_f1, per_class_report

CLASSES = ["ADVE", "Email", "Form", "Letter", "Memo",
           "News", "Note", "Report", "Resume", "Scientific"]
THRESH = 3   # same Hamming threshold as the dedup (D-030)


def main() -> None:
    run = REPO / "outputs" / "runs" / "0906_002054_gap3_seed0_first"
    pr = np.load(run / "test_predictions.npy", allow_pickle=True).item()
    y_true, y_pred = pr["y_true"], pr["y_pred"]

    imgs = np.load(REPO / "data" / "cache" / "images.npy", mmap_mode="r")
    sp = load_splits(REPO / "data" / "splits" / "splits.json")

    # Hash the CACHED tensors -- the representation the model consumes. Hashing the
    # originals is what caused the leak in the first place (D-043).
    ph = lambda i: imagehash.phash(Image.fromarray(np.asarray(imgs[i])))
    print(f"hashing {len(sp['train'])} train + {len(sp['test'])} test cached images ...")
    train_h = [ph(i) for i in sp["train"]]
    contaminated = np.array([min(ph(t) - x for x in train_h) <= THRESH
                             for t in sp["test"]])

    clean = ~contaminated
    out = {
        "n_test_total": int(len(y_true)),
        "n_contaminated": int(contaminated.sum()),
        "n_clean": int(clean.sum()),
        "hamming_threshold": THRESH,
        "raw": {"acc": float((y_true == y_pred).mean()),
                "macro_f1": macro_f1(y_true, y_pred)},
        "clean": {"acc": float((y_true[clean] == y_pred[clean]).mean()),
                  "macro_f1": macro_f1(y_true[clean], y_pred[clean]),
                  "per_class": per_class_report(y_true[clean], y_pred[clean], CLASSES)},
        "contaminated_acc": float((y_true[contaminated] == y_pred[contaminated]).mean()),
        "note": ("Contaminated = a test image within Hamming <= 3 of a training image "
                 "when both are hashed as the 256x256 CACHED tensors the model actually "
                 "consumes. The dedup that produced the split hashed the ORIGINAL scans, "
                 "which at ~9x downscaling is a different question (D-043)."),
    }
    p = REPO / "outputs" / "clean_metrics.json"
    p.write_text(json.dumps(out, indent=2))

    print(f"\n  test images        : {out['n_test_total']}")
    print(f"  contaminated       : {out['n_contaminated']} "
          f"({out['n_contaminated']/out['n_test_total']:.1%}), "
          f"accuracy on them {out['contaminated_acc']:.4f}")
    print(f"  clean              : {out['n_clean']}")
    print(f"  RAW   acc {out['raw']['acc']:.4f}  macro-F1 {out['raw']['macro_f1']:.4f}")
    print(f"  CLEAN acc {out['clean']['acc']:.4f}  macro-F1 {out['clean']['macro_f1']:.4f}")
    print(f"\n  saved -> {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
