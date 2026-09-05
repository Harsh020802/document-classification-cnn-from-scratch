"""Phase 1.2 steps 2-3 -- near-duplicate detection, then the stratified split.

Order matters (D-027): dedup runs BEFORE splitting so clusters can be kept inside a
single split, and the split runs BEFORE the cache build so normalization stats can be
computed on the training split only.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import imagehash
import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.data.dedup import cluster_ids, find_clusters
from src.data.splits import grouped_stratified_split, report_split, save_splits
from src.utils.seed import set_seed


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])

    ds = load_dataset(cfg["data"]["source"], cache_dir=str(REPO / cfg["data"]["raw_dir"]))["train"]
    labels = np.array(ds["label"])
    names = ds.features["label"].names
    n = len(ds)

    # ---- step 2: perceptual hashing -------------------------------------------
    thr = cfg["data"]["dedup"]["hamming_threshold"]
    print(f"phash over {n} images (Hamming <= {thr}) ...")
    hashes = [imagehash.phash(ds[i]["image"]) for i in tqdm(range(n), ncols=70)]

    rep = find_clusters(hashes, labels, threshold=thr)

    print(f"\n=== NEAR-DUPLICATE REPORT (phash, Hamming <= {thr}) ===")
    print(f"  images                    : {rep.n_images}")
    print(f"  multi-image clusters      : {len(rep.clusters)}")
    print(f"  images inside a cluster   : {rep.n_duplicated} ({rep.n_duplicated/n:.1%})")
    print(f"  cross-label clusters      : {len(rep.cross_label)}")

    if rep.clusters:
        sizes = [len(c) for c in rep.clusters]
        print(f"  cluster size min/med/max  : {min(sizes)} / {sorted(sizes)[len(sizes)//2]} / {max(sizes)}")
        print("\n  10 largest clusters:")
        for c in rep.clusters[:10]:
            labs = [names[labels[i]] for i in c]
            uniq = sorted(set(labs))
            flag = "  <-- CROSS-LABEL" if len(uniq) > 1 else ""
            print(f"    size {len(c):>3}  {uniq}{flag}")

    if rep.cross_label:
        print(f"\n  === CROSS-LABEL CLUSTERS ({len(rep.cross_label)}) ===")
        print("  These are near-identical images with DIFFERENT labels -- either label")
        print("  errors or genuinely ambiguous documents. The dataset has a known 11.7%")
        print("  label-error rate (arXiv 2412.13140), so both are expected.")
        for c in rep.cross_label[:15]:
            print(f"    idx {c[:6]}{'...' if len(c) > 6 else ''}  "
                  f"labels {[names[labels[i]] for i in c[:6]]}")

    np.save(REPO / cfg["data"]["splits_dir"] / "phash_groups.npy", cluster_ids(rep, n))

    # ---- step 3: split ---------------------------------------------------------
    groups = cluster_ids(rep, n)
    fr = cfg["data"]["split"]
    splits = grouped_stratified_split(
        labels, groups, (fr["train"], fr["val"], fr["test"]), seed=cfg["seed"]
    )

    print("\n=== STRATIFIED SPLIT (clusters kept intact) ===")
    report_split(splits, labels, names)

    # Leakage assertion: no group may straddle two splits.
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a >= b:
                continue
            shared = set(groups[splits[a]]) & set(groups[splits[b]])
            assert not shared, f"LEAK: {len(shared)} groups shared between {a} and {b}"
    print("\n  leakage check: no near-duplicate cluster spans two splits  OK")

    out = REPO / cfg["data"]["splits_dir"] / "splits.json"
    save_splits(splits, out)
    print(f"  saved -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
