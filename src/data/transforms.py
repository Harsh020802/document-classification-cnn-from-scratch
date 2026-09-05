"""Image preprocessing and augmentation.

Two stages, deliberately separated:

  1. CACHE-TIME (deterministic, done once): decode -> grayscale -> resize long side
     to 256 -> white-pad to 256x256. Cached as uint8, ~218 MB for 3,482 images.
  2. LOAD-TIME (random, every epoch): augment + crop to 224.

Caching the deterministic part and randomising at load is the point (D-009). If we
cached augmented images the model would see the same 2,400 variants every epoch and
the augmentation would do nothing.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# THE constant. Used by the pad AND every geometric transform's `fill`.
#
# torchvision's RandomRotation/RandomAffine default to fill=0 (black). Against
# white padding that paints maximum-contrast black wedges into the corners, and
# wedge area scales with |rotation angle|, which is resampled every epoch. That is
# random, class-irrelevant noise injected into the corners -- which are 4 of the 9
# gap3 spatial cells. It would bias the ablation AGAINST the spatial hypothesis.
#
# One constant, not two copies: two places holding the same magic number drift.
# (D-008)
# ---------------------------------------------------------------------------
PAD_VALUE = 255


def resize_and_pad(img: Image.Image, size: int = 256) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Resize long side to `size`, pad short side to square with white.

    Returns (padded_image, content_box) where content_box is (left, top, w, h) --
    where the real document sits inside the square canvas. Normalization stats are
    computed over content pixels only (D-010): including padding would skew the mean
    toward white by a per-image-varying amount and, worse, deflate the apparent
    variance of real content.

    The box is returned rather than a per-pixel mask because the mask is fully
    determined by these four numbers; storing it per pixel costs another 218 MB.

    Why pad rather than squash to square: pages are ~1:1.29 portrait, and a direct
    square resize compresses ~29%. The damage is not the distortion itself but that
    the distortion factor VARIES per image with source aspect ratio -- noise on
    exactly the geometric signal gap3 exists to read (D-007).
    """
    img = img.convert("L")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("L", (size, size), PAD_VALUE)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img, (left, top))
    return canvas, (left, top, nw, nh)


def build_train_transform(cfg: dict, mean: float, std: float) -> T.Compose:
    """Augmentation for training, applied to the cached 256x256 uint8 tensor.

    NO horizontal flip: mirrored text is not a document. Standard for CIFAR,
    actively wrong here.

    Every geometric op passes fill=PAD_VALUE and runs in pixel space BEFORE
    normalization, so "white" is genuinely white.
    """
    aug = cfg["augment"]
    size = cfg["data"]["input_size"]
    return T.Compose([
        # Random crop 256 -> 224 gives +/-32px translation jitter for free. Real
        # scans vary in page position, so this is realistic variation, not synthetic.
        T.RandomCrop(size),
        T.RandomAffine(
            degrees=aug["rotation_degrees"],
            translate=(aug["translate"], aug["translate"]),
            scale=tuple(aug["scale"]),
            fill=PAD_VALUE,          # <-- D-008. NOT the default 0.
        ),
        T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
        T.ToTensor(),
        T.Normalize(mean=[mean], std=[std]),
    ])


def build_eval_transform(cfg: dict, mean: float, std: float) -> T.Compose:
    """Deterministic pipeline for val/test.

    Center crop, not random: a random crop at eval time would make the metric
    non-reproducible across runs, which breaks the four-arm ablation comparison.
    """
    size = cfg["data"]["input_size"]
    return T.Compose([
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[mean], std=[std]),
    ])


def content_mean_std(cache: np.ndarray, boxes: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """Mean/std over CONTENT pixels only, for the given (training) indices.

    Padding is white and the AMOUNT varies per image with aspect ratio. Including it
    pulls the mean toward white by a per-image-varying, class-irrelevant amount and
    -- worse -- deflates the apparent variance of real content, so dividing by that
    std under-normalizes the actual document pixels (D-010).
    """
    # Streamed as sum / sum-of-squares so we never materialise a giant array.
    n = 0
    s = 0.0
    ss = 0.0
    for i in idx:
        left, top, w, h = boxes[i]
        px = cache[i, top:top + h, left:left + w].astype(np.float64) / 255.0
        n += px.size
        s += px.sum()
        ss += (px ** 2).sum()
    mean = s / n
    return float(mean), float(np.sqrt(max(ss / n - mean ** 2, 0.0)))


def padded_mean_std(cache: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """Same but including padding -- computed only to print the two side by side."""
    px = cache[idx].astype(np.float64) / 255.0
    return float(px.mean()), float(px.std())
