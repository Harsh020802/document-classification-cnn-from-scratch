"""Phase 1.5 -- the three trivial baselines. Run BEFORE the CNN.

Without a floor, an accuracy number is uninterpretable: "68%" means nothing until you
know that predicting the majority class gets 17.8% and a linear model on raw pixels
gets X%.

  1. Majority class          -- predict Memo always
  2. Logistic regression 8x8   (64 features)   -- near-pure ink density by region
  3. Logistic regression 32x32 (1024 features)

Regularisation matters (D-013): 1024 features on 2,438 samples overfits badly at
sklearn defaults, and a weak baseline is worse than no baseline -- it hands the CNN a
win it did not earn. So: StandardScaler fit on TRAIN only, C tuned on VALIDATION,
class_weight='balanced' to mirror the CNN's weighted loss.

The 8x8 arm is diagnostic: if 32x32 barely beats it, both are using coarse ink
distribution, and the margin the CNN must clear to show it learned STRUCTURE rather
than DENSITY is the margin over that.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.splits import load_splits
from src.engine.metrics import macro_f1, per_class_report
from src.utils.seed import set_seed

CLASSES = ["ADVE", "Email", "Form", "Letter", "Memo",
           "News", "Note", "Report", "Resume", "Scientific"]


def downscale(cache: np.ndarray, idx: np.ndarray, size: int) -> np.ndarray:
    """Cached 256x256 uint8 -> flattened size x size floats in [0, 1]."""
    out = np.zeros((len(idx), size * size), dtype=np.float32)
    for k, i in enumerate(idx):
        im = Image.fromarray(np.asarray(cache[i]), mode="L").resize(
            (size, size), Image.BILINEAR)
        out[k] = np.asarray(im, dtype=np.float32).ravel() / 255.0
    return out


def majority_baseline(y_tr: np.ndarray, y_te: np.ndarray) -> dict:
    cls = int(np.bincount(y_tr, minlength=10).argmax())
    pred = np.full_like(y_te, cls)
    return {
        "name": "majority class",
        "detail": f"always predict {CLASSES[cls]}",
        "acc": float((pred == y_te).mean()),
        "macro_f1": macro_f1(y_te, pred),
        "C": None,
    }


def logreg_baseline(cache, splits, y, size: int, seed: int) -> dict:
    Xtr = downscale(cache, splits["train"], size)
    Xva = downscale(cache, splits["val"], size)
    Xte = downscale(cache, splits["test"], size)
    ytr, yva, yte = y[splits["train"]], y[splits["val"]], y[splits["test"]]

    # Fit the scaler on TRAIN ONLY. Fitting on all splits would leak test statistics.
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    best = (-1.0, None, None)
    print(f"    tuning C on the VALIDATION split ({size}x{size} = {size*size} features):")
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                 class_weight="balanced", random_state=seed)
        clf.fit(Xtr, ytr)
        f1 = macro_f1(yva, clf.predict(Xva))
        acc = float((clf.predict(Xva) == yva).mean())
        flag = ""
        if f1 > best[0]:
            best = (f1, C, clf)
            flag = "  <-- best"
        print(f"      C={C:<7} val macro-F1 {f1:.4f}  val acc {acc:.4f}{flag}")

    _, C, clf = best
    pred = clf.predict(Xte)
    return {
        "name": f"logreg {size}x{size}",
        "detail": f"{size*size} features, StandardScaler(train), C tuned on val",
        "acc": float((pred == yte).mean()),
        "macro_f1": macro_f1(yte, pred),
        "C": C,
        "per_class": per_class_report(yte, pred, CLASSES),
    }


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])

    cache_dir = REPO / cfg["data"]["cache_dir"]
    cache = np.load(cache_dir / "images.npy", mmap_mode="r")
    y = np.load(cache_dir / "labels.npy")
    splits = load_splits(REPO / cfg["data"]["splits_dir"] / "splits.json")
    print(f"train {len(splits['train'])}  val {len(splits['val'])}  "
          f"test {len(splits['test'])}\n")

    results = []
    t0 = time.perf_counter()

    print("  [1/3] majority class")
    results.append(majority_baseline(y[splits["train"]], y[splits["test"]]))

    for size in (8, 32):
        print(f"\n  [{2 if size == 8 else 3}/3] logistic regression {size}x{size}")
        results.append(logreg_baseline(cache, splits, y, size, cfg["seed"]))

    print(f"\n{'=' * 78}\nBASELINE RESULTS -- test split (n={len(splits['test'])})\n{'=' * 78}")
    print(f"  {'model':<22}{'features':>10}{'C':>8}{'test acc':>11}{'macro-F1':>11}")
    print("  " + "-" * 62)
    for r in results:
        feats = {"majority class": "-", "logreg 8x8": "64", "logreg 32x32": "1024"}[r["name"]]
        c = f"{r['C']}" if r["C"] is not None else "-"
        print(f"  {r['name']:<22}{feats:>10}{c:>8}{r['acc']:>10.2%}{r['macro_f1']:>11.4f}")
    print("  " + "-" * 62)
    print(f"  {'CNN (to be measured)':<22}{'-':>10}{'-':>8}{'?':>10}{'?':>11}")

    a8 = next(r for r in results if r["name"] == "logreg 8x8")["acc"]
    a32 = next(r for r in results if r["name"] == "logreg 32x32")["acc"]
    print(f"\n  32x32 over 8x8: {a32 - a8:+.2%}")
    print("  If that margin is small, both baselines are using coarse ink distribution,")
    print("  and the CNN must clear 32x32 by a clear margin to show it learned")
    print("  STRUCTURE rather than DENSITY.")
    print(f"\n  elapsed {time.perf_counter() - t0:.1f}s")

    out = REPO / "outputs" / "baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  saved -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
