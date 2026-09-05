"""The Dataset class.

Reads from the uint8 cache built by scripts/04_build_cache.py rather than decoding
JPEGs. Source images are median 2386x2292 (D-029); decoding one per image per epoch
would starve the GPU, since the 1.2M-param model processes a batch in ~2 ms while a
decode of that size takes several.

The split between what is cached and what is random is the important design point
(D-009):

    CACHED (deterministic, once) : grayscale -> resize long side 256 -> white pad
    PER-EPOCH (random, at load)  : crop 224 + rotate/scale/jitter

Caching augmented images would defeat augmentation entirely -- the model would see the
same 2,438 variants every epoch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import build_eval_transform, build_train_transform


class DocumentCache:
    """The cached arrays, loaded once and shared by all three Dataset instances.

    Kept separate from Dataset so train/val/test do not each hold their own 218 MB
    copy -- which on an 8 GB machine would matter.
    """

    def __init__(self, cache_dir: Path, splits_path: Path, stats_path: Path) -> None:
        self.images = np.load(cache_dir / "images.npy", mmap_mode="r")
        self.labels = np.load(cache_dir / "labels.npy")
        self.boxes = np.load(cache_dir / "boxes.npy")
        self.splits = {k: np.array(v, dtype=np.int64)
                       for k, v in json.loads(splits_path.read_text()).items()}
        stats = json.loads(stats_path.read_text())
        self.mean, self.std = stats["mean"], stats["std"]


class DocumentDataset(Dataset):
    """One split of Tobacco-3482.

    Args:
        cache: the shared DocumentCache.
        split: "train" | "val" | "test".
        cfg: parsed config dict.
        train_mode: apply random augmentation. Defaults to (split == "train").
            Kept as an explicit argument so the timing experiment can measure the
            augmentation cost by disabling it without changing the split.
    """

    def __init__(self, cache: DocumentCache, split: str, cfg: dict,
                 train_mode: bool | None = None) -> None:
        self.cache = cache
        self.indices = cache.splits[split]
        self.split = split
        train_mode = (split == "train") if train_mode is None else train_mode
        self.transform = (build_train_transform if train_mode else build_eval_transform)(
            cfg, cache.mean, cache.std
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        idx = int(self.indices[i])
        # np.asarray() forces a real copy out of the memmap; PIL needs contiguous
        # memory and we must not hand a view to a transform that may write in place.
        arr = np.asarray(self.cache.images[idx])
        img = Image.fromarray(arr, mode="L")
        return self.transform(img), int(self.cache.labels[idx])

    @property
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.cache.labels[self.indices], minlength=10)


def class_weights(counts: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss.

    The dataset is 5.17x imbalanced (Memo 620, Resume 120). Unweighted, the loss is
    dominated by the large classes and the model can score respectably on accuracy
    while ignoring Resume entirely -- which is also why macro-F1, not accuracy, is the
    checkpoint-selection metric.

    Normalised to mean 1 so the overall loss scale stays comparable to unweighted
    training; otherwise the effective learning rate would shift with the weighting.
    """
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def build_datasets(cfg: dict, repo: Path) -> tuple[DocumentDataset, DocumentDataset, DocumentDataset]:
    cache = DocumentCache(
        repo / cfg["data"]["cache_dir"],
        repo / cfg["data"]["splits_dir"] / "splits.json",
        repo / cfg["data"]["normalize"]["stats_file"],
    )
    return (
        DocumentDataset(cache, "train", cfg),
        DocumentDataset(cache, "val", cfg),
        DocumentDataset(cache, "test", cfg),
    )
