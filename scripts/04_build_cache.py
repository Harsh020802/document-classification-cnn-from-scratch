"""Phase 1.2 step 6 -- build the uint8 tensor cache and compute normalization stats.

Two things happen here, and the ORDER between them matters:

  1. Cache every image at 256x256 (deterministic: grayscale, resize long side, white
     pad) plus a validity mask marking content vs padding.
  2. Compute normalization mean/std over the TRAINING split only, CONTENT pixels only.

Why cache (D-009): source images are median 2386x2292. Decoding one per image per epoch
would starve the GPU -- the model is 1.2M params and processes a batch in ~2 ms, while a
JPEG decode of that size is several ms. The cache makes the dataloader an in-memory slice.

Why cache 256 and not 224: augmentation at load time would otherwise resample
already-downscaled pixels. Cache the deterministic part, randomise at load.

Why train-only stats (D-027): `normalize.compute_from: train`. Computing over all 3,482
images would leak val/test pixel statistics into the normalization applied to training data.

Why content-pixels-only (D-010): padding is white and the AMOUNT varies per image with
aspect ratio (measured range 0.68-1.64, D-029). Including it skews the mean toward white
by a per-image-varying amount and deflates the apparent variance of real content.
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
from datasets import load_dataset
from tqdm import tqdm

from src.data.splits import load_splits
from src.data.transforms import content_mean_std, padded_mean_std, resize_and_pad
from src.utils.seed import set_seed


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])
    size = cfg["data"]["cache_size"]
    cache_dir = REPO / cfg["data"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(cfg["data"]["source"], cache_dir=str(REPO / cfg["data"]["raw_dir"]))["train"]
    n = len(ds)

    imgs = np.zeros((n, size, size), dtype=np.uint8)
    # Content geometry as (left, top, w, h) per image, not a full per-pixel mask.
    # The mask is fully described by the paste rectangle; storing it per pixel would
    # be another 218 MB on disk to encode 4 numbers per image.
    boxes = np.zeros((n, 4), dtype=np.int16)

    print(f"caching {n} images at {size}x{size} ...")
    t0 = time.perf_counter()
    for i in tqdm(range(n), ncols=70):
        padded, box = resize_and_pad(ds[i]["image"], size)
        imgs[i] = np.asarray(padded, dtype=np.uint8)
        boxes[i] = box
    build_s = time.perf_counter() - t0

    labels = np.array(ds["label"], dtype=np.int64)
    np.save(cache_dir / "images.npy", imgs)
    np.save(cache_dir / "boxes.npy", boxes)
    np.save(cache_dir / "labels.npy", labels)

    mb = imgs.nbytes / 1024**2
    print(f"\n  built in {build_s:.1f}s")
    print(f"  images.npy : {mb:.1f} MB   (predicted 217.6 MB)")
    print(f"  boxes.npy  : {boxes.nbytes / 1024:.1f} KB   (content geometry, not a pixel mask)")

    content_px = (boxes[:, 2].astype(np.int64) * boxes[:, 3]).sum()
    pad_frac = 1.0 - content_px / (n * size * size)
    print(f"  padding    : {pad_frac:.1%} of all cached pixels")

    # ---- normalization stats: TRAIN split, CONTENT pixels only ----------------
    splits = load_splits(REPO / cfg["data"]["splits_dir"] / "splits.json")
    tr = splits["train"]

    c_mean, c_std = content_mean_std(imgs, boxes, tr)
    p_mean, p_std = padded_mean_std(imgs, tr)

    print(f"\n=== NORMALIZATION STATS (train split, n={len(tr)}) ===")
    print(f"  {'':<26}{'mean':>9}{'std':>9}")
    print("  " + "-" * 44)
    print(f"  {'content pixels only':<26}{c_mean:>9.4f}{c_std:>9.4f}   <- USED")
    print(f"  {'including white padding':<26}{p_mean:>9.4f}{p_std:>9.4f}   (comparison only)")
    print("  " + "-" * 44)
    print(f"  {'difference':<26}{p_mean - c_mean:>+9.4f}{p_std - c_std:>+9.4f}")
    print(f"\n  D-010: including padding shifts the mean {p_mean - c_mean:+.4f} toward white")
    print(f"         and changes std by {p_std - c_std:+.4f}.")

    stats = {
        "mean": c_mean, "std": c_std,
        "mean_with_padding": p_mean, "std_with_padding": p_std,
        "computed_on": "train", "n_train": int(len(tr)),
        "exclude_padding": True, "cache_size": size,
        "padding_fraction": float(pad_frac),
    }
    out = REPO / cfg["data"]["normalize"]["stats_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f"\n  saved -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
