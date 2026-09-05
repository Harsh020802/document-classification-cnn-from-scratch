"""Predict on every image sitting in inbox/ -- drop files there and run this.

  ./run.sh scripts/predict_inbox.py                 # everything in inbox/
  ./run.sh scripts/predict_inbox.py --dir samples   # or any other folder
  ./run.sh scripts/predict_inbox.py path/to/one.jpg # or a single explicit file

Prints the full probability distribution over all ten classes, not just the top-3, plus
a confidence read based on the model's measured calibration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F
from PIL import Image

from scripts.predict import (CLASSES, find_best_checkpoint, load_model,
                             load_norm_stats, preprocess)
from src.utils.device import device_report, get_device

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def confidence_read(p: float) -> str:
    """Interpretation grounded in the measured reliability curve (D-047).

    Bins and their measured accuracy on the 521-image test set:
      >=0.99  -> 99.3% correct   (133 predictions)
      0.90-99 -> 95.1% correct   (144)
      0.70-90 -> 90.8% correct   (109)
      0.50-70 -> 67.9% correct   (84)
      <0.50   -> 37.3% correct   (51)
    """
    if p >= 0.99:
        return "very high -- this bin is 99.3% correct on the test set"
    if p >= 0.90:
        return "high -- this bin is 95.1% correct"
    if p >= 0.70:
        return "moderate -- this bin is 90.8% correct"
    if p >= 0.50:
        return "low -- this bin is only 67.9% correct; treat as a guess"
    return "very low -- this bin is 37.3% correct, i.e. usually WRONG"


def truth_from_name(path: Path) -> str | None:
    """Files exported by scripts/export_samples.py encode the true label in the name."""
    for part in path.stem.split("_"):
        if part in CLASSES:
            return part
    return None


def predict_one(path: Path, model, mean, std, device) -> None:
    img = Image.open(path)
    x = preprocess(img, mean, std).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu()

    order = torch.argsort(probs, descending=True)
    top, conf = CLASSES[order[0]], float(probs[order[0]])
    truth = truth_from_name(path)

    print("\n" + "=" * 68)
    print(f"  {path.name}")
    print("=" * 68)
    print(f"  source image : {img.size[0]} x {img.size[1]} px, mode {img.mode}")
    print(f"  fed to model : 1 x 1 x 224 x 224  (long side -> 256, white pad, centre crop)")
    print()
    print(f"  PREDICTION   : {top}")
    print(f"  CONFIDENCE   : {conf:.4f}  ({conf * 100:.2f}%)")
    print(f"  reliability  : {confidence_read(conf)}")
    if truth:
        ok = top == truth
        print(f"  TRUE LABEL   : {truth}   ->  {'CORRECT' if ok else 'WRONG'}")
    print()
    print("  full distribution over all 10 classes:")
    for r, i in enumerate(order, 1):
        c, p = CLASSES[i], float(probs[i])
        bar = "#" * max(int(p * 44), 1 if p >= 0.001 else 0)
        mark = ""
        if truth and c == truth:
            mark = "  <-- true label"
        elif r == 1:
            mark = "  <-- predicted"
        print(f"    {r:>2}. {c:<12}{p:>9.4f}  {bar}{mark}")

    margin = float(probs[order[0]] - probs[order[1]])
    print()
    print(f"  margin over 2nd place ({CLASSES[order[1]]}): {margin:.4f}")
    if margin < 0.15:
        print("    -> narrow. The model is genuinely torn between these two.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=None,
                    help="a single image, or omit to use the folder")
    ap.add_argument("--dir", default="inbox", help="folder to scan (default: inbox)")
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args()

    if a.path:
        files = [Path(a.path)]
    else:
        d = REPO / a.dir
        if not d.exists():
            sys.exit(f"{d} does not exist")
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in EXTS)
        if not files:
            sys.exit(f"no images in {d}/ -- drop a .jpg or .png in there and re-run")

    device = get_device()
    ckpt = find_best_checkpoint(a.checkpoint)
    model, ck = load_model(ckpt, device)
    mean, std = load_norm_stats()

    print(f"model      : {ckpt.relative_to(REPO)}")
    print(f"  {ck['arch']['head']['type']} head, epoch {ck['epoch']}, "
          f"val macro-F1 {ck['monitor_best']:.4f}")
    print(f"device     : {device_report(device)}")
    print(f"normalize  : mean {mean:.4f}  std {std:.4f}")
    print(f"images     : {len(files)} found in {a.dir if not a.path else 'given path'}/")

    for f in files:
        predict_one(f, model, mean, std, device)
    print()


if __name__ == "__main__":
    main()
