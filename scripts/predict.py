"""Single-image inference.

  PYTHONPATH=. python scripts/predict.py path/to/scan.jpg
  PYTHONPATH=. python scripts/predict.py --test-index 0        # pull one from the test split

THE THING THAT MATTERS HERE is that preprocessing matches training EXACTLY. A mismatch
does not raise -- it silently degrades accuracy and looks like the model got worse, which
is the worst kind of bug to chase. Specifically:

  * long-side resize to 256, white pad to square (PAD_VALUE=255), then CENTER crop to 224
    -- the eval path, not the train path. A random crop here would make predictions
    non-reproducible for the same file.
  * normalization uses the TRAIN-SPLIT, CONTENT-PIXEL-ONLY statistics from
    outputs/norm_stats.json: mean 0.9351, std 0.1726. NOT the padding-inflated pair
    (0.9502 / 0.1535). Using the padded std would under-normalise real content by ~11%.
    The file stores both; this script reads `mean`/`std`, and asserts the file was
    written with exclude_padding=True so the wrong pair cannot be picked up silently.

The architecture is reconstructed from the arch args stored IN the checkpoint, not
hardcoded -- otherwise this breaks the moment the ablation produces four different heads
(D-003: the checkpoint stores the full config dict, not just a class name).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from src.data.transforms import resize_and_pad
from src.models.cnn import DocCNN
from src.utils.device import device_report, get_device

CLASSES = ["ADVE", "Email", "Form", "Letter", "Memo",
           "News", "Note", "Report", "Resume", "Scientific"]


def find_best_checkpoint(explicit: str | None) -> Path:
    """The best.pt with the highest monitor_best (= max val_macro_f1) across all runs."""
    if explicit:
        return Path(explicit)
    cands = sorted((REPO / "outputs" / "runs").glob("*/best.pt"))
    if not cands:
        sys.exit("no checkpoint found -- run scripts/09_train.py first")
    scored = []
    for c in cands:
        try:
            mb = torch.load(c, map_location="cpu", weights_only=False)["monitor_best"]
            scored.append((float(mb), c))
        except Exception:
            continue
    if not scored:
        sys.exit("no readable checkpoint found")
    scored.sort(reverse=True)
    return scored[0][1]


def load_model(ckpt_path: Path, device: torch.device) -> tuple[DocCNN, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ck["arch"]
    # Rebuild from the STORED args, not hardcoded values.
    model = DocCNN(
        in_channels=arch["in_channels"],
        block_channels=tuple(arch["block_channels"]),
        num_classes=10,
        dropout=arch["dropout"],
        head_type=arch["head"]["type"],
        hidden_dim=arch["head"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()   # dropout OFF; BatchNorm uses running stats, not this batch's.
    return model, ck


def load_norm_stats() -> tuple[float, float]:
    p = REPO / "outputs" / "norm_stats.json"
    if not p.exists():
        sys.exit("outputs/norm_stats.json missing -- run scripts/04_build_cache.py")
    s = json.loads(p.read_text())
    # Guard against silently picking up the padding-inflated statistics.
    assert s.get("exclude_padding") is True, \
        "norm_stats.json was written WITHOUT exclude_padding -- regenerate the cache"
    return float(s["mean"]), float(s["std"])


def preprocess(img: Image.Image, mean: float, std: float,
               cache_size: int = 256, input_size: int = 224) -> torch.Tensor:
    """Identical to the eval path in src/data/transforms.build_eval_transform."""
    padded, _ = resize_and_pad(img, cache_size)          # grayscale, long side 256, white pad
    tf = T.Compose([
        T.CenterCrop(input_size),                        # deterministic, NOT RandomCrop
        T.ToTensor(),
        T.Normalize(mean=[mean], std=[std]),
    ])
    return tf(padded).unsqueeze(0)                       # (1, 1, 224, 224)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="path to an image file")
    ap.add_argument("--test-index", type=int, default=None,
                    help="instead of a file, use the Nth image of the test split")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--topk", type=int, default=3)
    a = ap.parse_args()

    if a.image is None and a.test_index is None:
        ap.error("give an image path or --test-index")

    device = get_device()
    ckpt_path = find_best_checkpoint(a.checkpoint)
    model, ck = load_model(ckpt_path, device)
    mean, std = load_norm_stats()

    print(f"checkpoint : {ckpt_path.relative_to(REPO)}")
    print(f"  arch     : {ck['arch']['head']['type']} head, "
          f"blocks {ck['arch']['block_channels']}")
    print(f"  trained  : epoch {ck['epoch']}, best val macro-F1 {ck['monitor_best']:.4f}")
    print(f"device     : {device_report(device)}")
    print(f"normalize  : mean {mean:.4f}  std {std:.4f}  (train split, content pixels only)")

    truth = None
    if a.test_index is not None:
        from src.data.splits import load_splits
        idx = load_splits(REPO / "data" / "splits" / "splits.json")["test"]
        labels = np.load(REPO / "data" / "cache" / "labels.npy")
        ds_i = int(idx[a.test_index])
        truth = CLASSES[int(labels[ds_i])]
        from datasets import load_dataset
        import yaml
        cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
        hf = load_dataset(cfg["data"]["source"],
                          cache_dir=str(REPO / cfg["data"]["raw_dir"]))["train"]
        img = hf[ds_i]["image"]
        src = f"test split position {a.test_index} (dataset index {ds_i})"
    else:
        img = Image.open(a.image)
        src = str(a.image)

    print(f"input      : {src}")
    print(f"  original : {img.size[0]}x{img.size[1]} px, mode {img.mode}")

    x = preprocess(img, mean, std).to(device)
    print(f"  tensor   : {tuple(x.shape)}  range [{x.min():.3f}, {x.max():.3f}]")

    with torch.no_grad():          # no autograd graph: ~1.5 GB saved at batch 32, and
                                   # no risk of an accidental gradient step
        logits = model(x)          # RAW LOGITS -- the model applies no softmax
        probs = F.softmax(logits, dim=1)[0].cpu()

    k = min(a.topk, len(CLASSES))
    top = torch.topk(probs, k)
    print()
    print(f"  PREDICTION : {CLASSES[top.indices[0]]}   "
          f"confidence {top.values[0]:.4f} ({top.values[0]*100:.2f}%)")
    if truth is not None:
        ok = CLASSES[top.indices[0]] == truth
        print(f"  TRUE LABEL : {truth}   -> {'CORRECT' if ok else 'WRONG'}")
    print()
    print(f"  top-{k}:")
    for r in range(k):
        c, p = CLASSES[top.indices[r]], float(top.values[r])
        bar = "#" * int(p * 40)
        mark = "  <-- true" if truth and c == truth else ""
        print(f"    {r+1}. {c:<12}{p:>8.4f}  {bar}{mark}")


if __name__ == "__main__":
    main()
