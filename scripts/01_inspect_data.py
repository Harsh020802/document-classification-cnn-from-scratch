"""Phase 1.2 step 1 -- download Tobacco-3482 and inspect it.

Verifies the dataset matches what the plan assumed before we build anything on it:
3,482 images, 10 classes, and the 5.2x imbalance (Memo 619 ... Resume 120) that
drives the weighted loss and the macro-F1 choice.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml
from datasets import load_dataset


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    source = cfg["data"]["source"]
    raw_dir = REPO / cfg["data"]["raw_dir"]

    print(f"Loading {source} ...")
    ds = load_dataset(source, cache_dir=str(raw_dir))

    print("\n=== SPLITS AS SHIPPED ===")
    for name, split in ds.items():
        print(f"  {name:12s} {len(split):>6,} rows")
    print(f"  {'TOTAL':12s} {sum(len(s) for s in ds.values()):>6,}")

    # Work from whichever split(s) exist; we re-split ourselves anyway (D-027).
    split_name = "train" if "train" in ds else next(iter(ds))
    split = ds[split_name]

    print(f"\n=== FEATURES ({split_name}) ===")
    for key, feat in split.features.items():
        print(f"  {key:14s} {feat}")

    label_key = next((k for k in ("label", "labels") if k in split.features), None)
    if label_key is None:
        print("\n!! no label column found -- inspect features above")
        return

    names = split.features[label_key].names
    print(f"\n=== CLASSES ({len(names)}) ===")
    print(f"  {names}")

    counts = Counter()
    for s in ds.values():
        counts.update(s[label_key])

    print("\n=== CLASS DISTRIBUTION (all splits pooled) ===")
    total = sum(counts.values())
    print(f"  {'class':<16}{'n':>6}{'share':>9}")
    print("  " + "-" * 31)
    for idx, n in counts.most_common():
        print(f"  {names[idx]:<16}{n:>6}{n / total:>8.1%}")
    print("  " + "-" * 31)
    print(f"  {'TOTAL':<16}{total:>6}")

    hi, lo = max(counts.values()), min(counts.values())
    print(f"\n  imbalance: {hi} / {lo} = {hi / lo:.2f}x")

    print("\n=== EXPECTED vs ACTUAL ===")
    expected = {"total": 3482, "classes": 10, "imbalance": 5.16}
    checks = [
        ("total images", total, expected["total"]),
        ("num classes", len(names), expected["classes"]),
    ]
    for lbl, got, want in checks:
        print(f"  {lbl:<14} got {got:<8} expected {want:<8} {'OK' if got == want else 'MISMATCH'}")
    print(f"  {'imbalance':<14} got {hi / lo:<8.2f} expected ~{expected['imbalance']:<7.2f}")

    print("\n=== IMAGE PROPERTIES (first 200) ===")
    sizes, modes = [], Counter()
    for i in range(min(200, len(split))):
        img = split[i]["image"]
        sizes.append(img.size)
        modes[img.mode] += 1
    ws = [w for w, _ in sizes]
    hs = [h for _, h in sizes]
    ars = [h / w for w, h in sizes]
    print(f"  width   min/med/max : {min(ws)} / {sorted(ws)[len(ws)//2]} / {max(ws)}")
    print(f"  height  min/med/max : {min(hs)} / {sorted(hs)[len(hs)//2]} / {max(hs)}")
    print(f"  aspect (h/w)        : {min(ars):.2f} / {sorted(ars)[len(ars)//2]:.2f} / {max(ars):.2f}")
    print(f"  modes               : {dict(modes)}")
    print("\n  (plan assumed ~1:1.29 portrait -- D-007 pad-to-square)")


if __name__ == "__main__":
    main()
