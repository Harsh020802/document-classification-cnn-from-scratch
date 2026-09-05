"""Phase 1.2 steps 4-5 -- the two visual sanity checks.

These run BEFORE the cache build (D-027): if the padding is wrong, better to see it
now than after generating 218 MB.

Step 4 -- squashed vs padded (D-007). Shows what a direct 224x224 resize does to a
portrait page versus resize-long-side-and-pad.

Step 5 -- augmented samples (D-008). THE fill=255 check. A fill=0 default paints black
wedges into the corners of every rotated image, and corners are 4 of the 9 gap3 spatial
cells. This bug is invisible in a confusion matrix and obvious in two seconds of looking
at the image. The panel deliberately includes a fill=0 row so the difference is visible
side by side rather than asserted.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from datasets import load_dataset
from PIL import Image
from torchvision import transforms as T

from src.data.transforms import PAD_VALUE, resize_and_pad
from src.utils.seed import set_seed


def squash_vs_pad(ds, labels, names, out: Path, n: int = 4) -> None:
    """Step 4: side-by-side of the two resize strategies."""
    # Pick images spanning the aspect-ratio range, since that is the whole point.
    idx = list(range(0, len(ds), max(1, len(ds) // 60)))[:60]
    ars = [(i, ds[i]["image"].size[1] / ds[i]["image"].size[0]) for i in idx]
    ars.sort(key=lambda t: t[1])
    picks = [ars[0][0], ars[len(ars) // 3][0], ars[2 * len(ars) // 3][0], ars[-1][0]][:n]

    fig, axes = plt.subplots(3, n, figsize=(3.1 * n, 9.6))
    for col, i in enumerate(picks):
        img = ds[i]["image"].convert("L")
        w, h = img.size
        ar = h / w
        squashed = img.resize((224, 224), Image.BILINEAR)
        padded, _ = resize_and_pad(img, 224)

        for row, (im, title) in enumerate([
            (img, f"original {w}x{h}\naspect h/w = {ar:.2f}"),
            (squashed, "SQUASHED to 224x224\n(varies per document)"),
            (padded, "resize long side + white pad\n(geometry preserved)"),
        ]):
            ax = axes[row, col]
            ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{names[labels[i]]}\n{title}" if row == 0 else title, fontsize=8)
            ax.axis("off")

    fig.suptitle("D-007: squash vs pad-to-square  (aspect ratios span 0.68-1.64)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


def augmentation_panel(ds, labels, names, cfg, out: Path) -> None:
    """Step 5: THE fill check. Top row fill=0 (wrong), bottom row fill=255 (correct)."""
    aug = cfg["augment"]
    i = 7
    img = ds[i]["image"].convert("L")
    cached, _ = resize_and_pad(img, cfg["data"]["cache_size"])  # 256x256, as cached

    def make(fill: int, seed: int):
        set_seed(seed)
        return T.Compose([
            T.RandomCrop(cfg["data"]["input_size"]),
            T.RandomAffine(
                degrees=aug["rotation_degrees"],
                translate=(aug["translate"], aug["translate"]),
                scale=tuple(aug["scale"]),
                fill=fill,
            ),
            T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
        ])(cached)

    ncol = 5
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.6))
    for c in range(ncol):
        for r, (fill, lbl) in enumerate([(0, "fill=0  (torchvision DEFAULT - WRONG)"),
                                         (PAD_VALUE, f"fill={PAD_VALUE}  (ours - CORRECT)")]):
            a = make(fill, 100 + c)
            ax = axes[r, c]
            ax.imshow(np.asarray(a), cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(lbl, fontsize=9)
                ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            # Ring the corners: that is where a fill bug shows.
            for (x, y) in [(0, 0), (224 - 46, 0), (0, 224 - 46), (224 - 46, 224 - 46)]:
                ax.add_patch(plt.Rectangle((x, y), 46, 46, fill=False,
                                           ec="red", lw=1.0, ls=":"))
    axes[0, 0].set_ylabel("fill=0\nWRONG", fontsize=9, color="crimson")
    axes[1, 0].set_ylabel(f"fill={PAD_VALUE}\nCORRECT", fontsize=9, color="green")
    fig.suptitle(
        f"D-008: augmentation fill value  |  class = {names[labels[i]]}  |  "
        "red boxes = corner regions (4 of the 9 gap3 cells)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text())
    set_seed(cfg["seed"])
    ds = load_dataset(cfg["data"]["source"], cache_dir=str(REPO / cfg["data"]["raw_dir"]))["train"]
    labels = np.array(ds["label"])
    names = ds.features["label"].names
    figs = REPO / "outputs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print("step 4: squashed vs padded ...")
    squash_vs_pad(ds, labels, names, figs / "01_squash_vs_pad.png")
    print("step 5: augmentation fill check ...")
    augmentation_panel(ds, labels, names, cfg, figs / "02_augmentation_fill.png")


if __name__ == "__main__":
    main()
