"""Part 2 -- rebuild the split using CACHE-SPACE dedup, then recompute baselines.

D-043: the original dedup hashed the ORIGINAL scans (median 2386x2292) while the model
consumes 256x256 cached tensors. At ~9x downscaling those are different questions, and 13
test images turned out to be near-duplicates of training images as the model sees them.

This rebuilds the split hashing the CACHE, which is the representation that governs
leakage. The old split is preserved as splits_origspace.json.
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import imagehash
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from src.data.dedup import cluster_ids, find_clusters
from src.data.splits import grouped_stratified_split, report_split, save_splits
from src.utils.seed import set_seed

CLASSES = ["ADVE", "Email", "Form", "Letter", "Memo",
           "News", "Note", "Report", "Resume", "Scientific"]


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])
    sd = REPO / cfg["data"]["splits_dir"]

    imgs = np.load(REPO / cfg["data"]["cache_dir"] / "images.npy", mmap_mode="r")
    labels = np.load(REPO / cfg["data"]["cache_dir"] / "labels.npy")
    n = len(labels)

    thr = cfg["data"]["dedup"]["hamming_threshold"]
    print(f"hashing {n} CACHED 256x256 images (Hamming <= {thr}) ...", flush=True)
    hashes = [imagehash.phash(Image.fromarray(np.asarray(imgs[i])))
              for i in tqdm(range(n), ncols=70)]

    rep = find_clusters(hashes, labels, threshold=thr)
    print("\n=== CACHE-SPACE DEDUP (original-space figures in brackets) ===")
    print(f"  multi-image clusters    : {len(rep.clusters)}   [31]")
    print(f"  images in a cluster     : {rep.n_duplicated} "
          f"({rep.n_duplicated/n:.1%})   [116, 3.3%]")
    print(f"  cross-label clusters    : {len(rep.cross_label)}   [1]")
    if rep.clusters:
        s = [len(c) for c in rep.clusters]
        print(f"  cluster size min/med/max: {min(s)} / {sorted(s)[len(s)//2]} / "
              f"{max(s)}   [2/2/25]")

    groups = cluster_ids(rep, n)
    fr = cfg["data"]["split"]
    splits = grouped_stratified_split(labels, groups,
                                      (fr["train"], fr["val"], fr["test"]),
                                      seed=cfg["seed"])
    print("\n=== SPLIT (cache-space clusters kept intact) ===")
    report_split(splits, labels, CLASSES)

    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                assert not (set(groups[splits[a]]) & set(groups[splits[b]])), f"{a}/{b} leak"
    print("\n  group-leakage check: OK")

    tr_h = [hashes[i] for i in splits["train"]]
    close = sum(1 for t in splits["test"] if min(hashes[t] - x for x in tr_h) <= thr)
    print(f"  test images within Hamming <= {thr} of a train image: {close}   "
          f"[was 13 on the old split]")

    if (sd / "splits.json").exists() and not (sd / "splits_origspace.json").exists():
        shutil.copy(sd / "splits.json", sd / "splits_origspace.json")
        shutil.copy(sd / "phash_groups.npy", sd / "phash_groups_origspace.npy")
        print("  old split preserved as splits_origspace.json")

    save_splits(splits, sd / "splits.json")
    np.save(sd / "phash_groups.npy", groups)
    (REPO / "outputs" / "dedup_cachespace.json").write_text(json.dumps({
        "space": "cache_256", "threshold": thr, "n_clusters": len(rep.clusters),
        "n_in_cluster": int(rep.n_duplicated), "n_cross_label": len(rep.cross_label),
        "test_near_train": int(close),
        "sizes": {k: int(len(v)) for k, v in splits.items()},
    }, indent=2))
    print("  saved -> data/splits/splits.json (cache-space)")


if __name__ == "__main__":
    main()
