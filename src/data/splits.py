"""Stratified train/val/test split, respecting near-duplicate clusters.

Two requirements that interact:

  1. STRATIFIED (D-001): the dataset is 5.2x imbalanced (Memo 619 ... Resume 120).
     A plain random split could hand us ~4 resumes in validation out of 120 total,
     which would make per-class validation metrics meaningless. Both reference repos
     use unstratified random splits; bentrevett's copy.deepcopy trick in particular
     is unacceptable here.

  2. GROUPED BY NEAR-DUPLICATE CLUSTER (D-011): if the same scanned document appears
     in both train and test, the test score is inflated. That is the >85% "check for
     leakage" scenario in the target table.

These pull against each other -- you cannot in general satisfy exact stratification
AND keep arbitrary groups intact. The approach below prioritises group integrity
(leakage is a correctness bug; slight stratification drift is a nuisance) and then
reports the achieved per-class proportions so any drift is visible rather than assumed
away.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def grouped_stratified_split(
    labels: np.ndarray,
    groups: np.ndarray,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split indices into train/val/test.

    Each group (near-duplicate cluster) lands entirely in one split. Groups are
    assigned greedily, class by class, to whichever split is furthest below its
    target for that class -- which keeps proportions close without ever splitting a
    group.

    Args:
        labels: (N,) integer class labels.
        groups: (N,) group id per image; singletons have unique ids.
        fractions: (train, val, test), must sum to 1.
        seed: shuffle seed.

    Returns:
        {"train": idx, "val": idx, "test": idx} as sorted int arrays.
    """
    assert abs(sum(fractions) - 1.0) < 1e-9, "fractions must sum to 1"
    rng = np.random.default_rng(seed)
    n = len(labels)
    names = ("train", "val", "test")

    # Group -> member indices. A group's class is its majority label (cross-label
    # clusters exist and are flagged separately by the dedup report).
    order: dict[int, list[int]] = {}
    for i, g in enumerate(groups):
        order.setdefault(int(g), []).append(i)

    group_ids = list(order.keys())
    rng.shuffle(group_ids)

    n_classes = int(labels.max()) + 1
    # target[s][c] = how many images of class c split s should end up with
    class_totals = np.bincount(labels, minlength=n_classes)
    target = {s: class_totals * f for s, f in zip(names, fractions)}
    have = {s: np.zeros(n_classes) for s in names}
    assigned: dict[str, list[int]] = {s: [] for s in names}

    # Largest groups first: they are the least flexible, so place them while there
    # is still slack to absorb them.
    group_ids.sort(key=lambda g: -len(order[g]))

    for g in group_ids:
        members = order[g]
        counts = np.bincount(labels[members], minlength=n_classes)
        # Deficit = how far below target each split is for the classes in this group.
        # Assign to whichever split is hungriest.
        deficits = {s: float(((target[s] - have[s]) * counts).sum()) for s in names}
        best = max(names, key=lambda s: deficits[s])
        assigned[best].extend(members)
        have[best] += counts

    out = {s: np.array(sorted(assigned[s]), dtype=np.int64) for s in names}
    assert sum(len(v) for v in out.values()) == n, "split lost images"
    assert len(set().union(*(set(v.tolist()) for v in out.values()))) == n, "overlap"
    return out


def report_split(splits: dict[str, np.ndarray], labels: np.ndarray, class_names: list[str]) -> None:
    """Print achieved per-class proportions so stratification drift is visible."""
    n = sum(len(v) for v in splits.values())
    print(f"\n{'class':<16}{'total':>7}{'train':>8}{'val':>7}{'test':>7}   {'tr%':>6}{'va%':>6}{'te%':>6}")
    print("-" * 70)
    for c, name in enumerate(class_names):
        tot = int((labels == c).sum())
        row = [int((labels[splits[s]] == c).sum()) for s in ("train", "val", "test")]
        pct = [r / tot * 100 if tot else 0 for r in row]
        print(f"{name:<16}{tot:>7}{row[0]:>8}{row[1]:>7}{row[2]:>7}   "
              f"{pct[0]:>5.1f}%{pct[1]:>5.1f}%{pct[2]:>5.1f}%")
    print("-" * 70)
    sizes = [len(splits[s]) for s in ("train", "val", "test")]
    print(f"{'TOTAL':<16}{n:>7}{sizes[0]:>8}{sizes[1]:>7}{sizes[2]:>7}   "
          f"{sizes[0]/n*100:>5.1f}%{sizes[1]/n*100:>5.1f}%{sizes[2]/n*100:>5.1f}%")
    print("\ntarget: 70.0% / 15.0% / 15.0%  (drift comes from keeping duplicate "
          "clusters intact)")


def save_splits(splits: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.tolist() for k, v in splits.items()}, indent=2))


def load_splits(path: Path) -> dict[str, np.ndarray]:
    raw = json.loads(path.read_text())
    return {k: np.array(v, dtype=np.int64) for k, v in raw.items()}
