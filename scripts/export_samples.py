"""Export images from the dataset as real files you can open and look at.

The dataset lives inside HuggingFace Arrow shards, so there is nothing to browse in
Finder. This writes actual JPEGs to a folder.

  PYTHONPATH=. python scripts/export_samples.py                  # 20 random test images
  PYTHONPATH=. python scripts/export_samples.py --n 50           # 50 of them
  PYTHONPATH=. python scripts/export_samples.py --class Resume   # only one class
  PYTHONPATH=. python scripts/export_samples.py --split all      # sample from everything

Filenames are  <split>_<position>_<TRUELABEL>_<datasetindex>.jpg
so the answer is in the name -- pick one, run predict.py on it, and compare.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import yaml
from datasets import load_dataset

from src.data.splits import load_splits

CLASSES = ["ADVE", "Email", "Form", "Letter", "Memo",
           "News", "Note", "Report", "Resume", "Scientific"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="how many to export")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--class", dest="cls", default=None, choices=CLASSES,
                    help="restrict to one class")
    ap.add_argument("--out", default="samples", help="output directory")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the random choice; omit for a different set each run")
    a = ap.parse_args()

    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    ds = load_dataset(cfg["data"]["source"],
                      cache_dir=str(REPO / cfg["data"]["raw_dir"]))["train"]
    labels = np.load(REPO / cfg["data"]["cache_dir"] / "labels.npy")
    splits = load_splits(REPO / cfg["data"]["splits_dir"] / "splits.json")

    if a.split == "all":
        pool = np.arange(len(ds))
        pos_of = {int(i): ("?", -1) for i in pool}
    else:
        pool = splits[a.split]
        pos_of = {int(idx): (a.split, p) for p, idx in enumerate(pool)}

    if a.cls is not None:
        want = CLASSES.index(a.cls)
        pool = np.array([i for i in pool if labels[i] == want])
        if len(pool) == 0:
            sys.exit(f"no {a.cls} images in the {a.split} split")

    rng = np.random.default_rng(a.seed)
    pick = rng.choice(pool, size=min(a.n, len(pool)), replace=False)

    out = REPO / a.out
    out.mkdir(exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()

    print(f"exporting {len(pick)} images from the {a.split} split -> {out}/\n")
    for i in sorted(pick):
        i = int(i)
        split, pos = pos_of[i]
        name = f"{split}_{pos:03d}_{CLASSES[labels[i]]}_{i}.jpg"
        ds[i]["image"].convert("L").save(out / name, quality=92)
        print(f"  {name}")

    print(f"\nOpen the folder:   open {a.out}")
    print("Filename format:   <split>_<position>_<TRUE LABEL>_<dataset index>.jpg")
    print("\nThen predict on one:")
    ex = sorted(pick)[0]
    split, pos = pos_of[int(ex)]
    print(f"  PYTHONPATH=. python scripts/predict.py "
          f"{a.out}/{split}_{pos:03d}_{CLASSES[labels[int(ex)]]}_{int(ex)}.jpg")


if __name__ == "__main__":
    main()
