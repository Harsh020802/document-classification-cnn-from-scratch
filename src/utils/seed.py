"""Reproducibility.

IMPORTANT CAVEAT (this is why the module has a docstring instead of being three
lines inline):

Both reference repos set `torch.backends.cudnn.deterministic = True` and treat the
seeding block as sufficient for reproducibility. On Apple Silicon that line is a
NO-OP -- cuDNN is an NVIDIA library and there is no cuDNN on Metal. PyTorch makes
no bit-determinism guarantee for the MPS backend.

So seeding here reduces run-to-run variation but does NOT guarantee identical
results. That is exactly why the ablation protocol measures a seed-variance floor
empirically (gap3 x 3 seeds, reported as an observed range) instead of assuming
seeding removes the problem. See notes/ for the full reasoning.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that affects training.

    Args:
        seed: the seed value.
        deterministic: request deterministic kernels where the backend supports it.
            Has real effect on CUDA; effectively inert on MPS (see module docstring).
    """
    # PYTHONHASHSEED affects set/dict iteration order, which can influence any
    # code path that iterates an unordered collection to build data.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)          # torchvision transforms use Python's `random`
    np.random.seed(seed)       # sklearn splits, phash clustering
    torch.manual_seed(seed)    # weight init, dropout masks, dataloader shuffling

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Real effect on CUDA only. Kept so the same code is reproducible if the
        # ablation is later re-run on Kaggle, where cuDNN does exist.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init fn, so each worker gets a distinct but derived seed.

    Currently unused (we plan num_workers=0 once the cache is in RAM), but kept
    because the timing experiment in Phase 1.2 measures num_workers in {0, 2} and
    the 2-worker arm needs this to stay reproducible.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
