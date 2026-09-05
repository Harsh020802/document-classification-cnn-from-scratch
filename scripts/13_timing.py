"""Phase 1.2 step 7 -- the timing experiment, deferred from Saturday.

Measures per-epoch time across two factors:
  cache      : the 256x256 uint8 tensor cache vs decoding source JPEGs each epoch
  num_workers: 0 vs 2

Two predictions being tested:
  1. The cache matters MORE than the Phase 0.4 estimate assumed, because source images
     are median 2386x2292 -- ~7x the pixels that estimate was based on.
  2. num_workers=0 should win once the cache is in RAM, because there is no blocking
     I/O left for worker processes to hide, and each forked worker costs memory on an
     8 GB machine. Predicted, not assumed -- hence this measurement.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import build_datasets, class_weights
from src.data.transforms import build_train_transform, resize_and_pad
from src.models.cnn import build_model
from src.utils.device import device_report, get_device
from src.utils.seed import set_seed


class UncachedDataset(Dataset):
    """Decodes the source JPEG on every __getitem__ -- the no-cache arm."""

    def __init__(self, hf_ds, indices, cfg, mean, std):
        self.ds, self.idx = hf_ds, indices
        self.size = cfg["data"]["cache_size"]
        self.tf = build_train_transform(cfg, mean, std)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        row = self.ds[int(self.idx[i])]
        img, _ = resize_and_pad(row["image"], self.size)
        return self.tf(img), int(row["label"])


def time_epoch(loader, model, criterion, optimizer, device, max_batches=None) -> float:
    model.train()
    t0 = time.perf_counter()
    for n, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        if max_batches and n + 1 >= max_batches:
            break
    if device.type == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])
    device = get_device()
    print(f"device: {device_report(device)}")

    train_ds, _, _ = build_datasets(cfg, REPO)
    n_train = len(train_ds)
    bs = cfg["data"]["loader"]["batch_size"]
    # Time a fixed number of batches and scale, so the uncached arm does not take
    # 20 minutes to measure.
    NB = 20
    full = int(np.ceil(n_train / bs))
    print(f"train {n_train} images, batch {bs} -> {full} batches/epoch")
    print(f"timing {NB} batches per configuration, scaled to a full epoch\n")

    model = build_model(cfg).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights(train_ds.class_counts).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    hf = load_dataset(cfg["data"]["source"], cache_dir=str(REPO / cfg["data"]["raw_dir"]))["train"]
    stats = __import__("json").loads((REPO / cfg["data"]["normalize"]["stats_file"]).read_text())
    uncached = UncachedDataset(hf, train_ds.indices, cfg, stats["mean"], stats["std"])

    results = {}
    for cache_on, ds in (("cached", train_ds), ("no cache", uncached)):
        for nw in (0, 2):
            loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw)
            time_epoch(loader, model, crit, opt, device, max_batches=3)   # warm up
            dt = time_epoch(loader, model, crit, opt, device, max_batches=NB)
            per_epoch = dt / NB * full
            results[(cache_on, nw)] = per_epoch
            print(f"  {cache_on:<10} num_workers={nw}: "
                  f"{dt/NB*1000:>7.1f} ms/batch -> {per_epoch:>7.1f} s/epoch")

    print(f"\n  {'':<12}{'workers=0':>12}{'workers=2':>12}")
    print("  " + "-" * 36)
    for c in ("cached", "no cache"):
        print(f"  {c:<12}{results[(c,0)]:>11.1f}s{results[(c,2)]:>11.1f}s")
    print()
    speedup = results[("no cache", 0)] / results[("cached", 0)]
    print(f"  cache speedup (workers=0): {speedup:.1f}x")
    best = min(results, key=results.get)
    print(f"  fastest configuration    : {best[0]}, num_workers={best[1]} "
          f"({results[best]:.1f} s/epoch)")
    print(f"  actual measured run      : 36.8 s/epoch (includes validation)")

    (REPO / "outputs" / "timing.json").write_text(__import__("json").dumps(
        {f"{c}_workers{w}": v for (c, w), v in results.items()}, indent=2))
    print(f"\n  saved -> outputs/timing.json")


if __name__ == "__main__":
    main()
