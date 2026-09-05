"""Checkpoint save/load.

Format adopted from victoresque's pytorch-template, with one change (D-003): the arch
entry stores the full CONFIG DICT, not a bare class name. Every ablation arm is a
`DocCNN`, so victoresque's class-name comparison would never fire and the four arms
would be indistinguishable on disk. Storing the args makes the check meaningful and the
checkpoint self-describing -- which is what lets `outputs/checkpoints/best.pt` still be
interpretable in three months.

What goes in and why:

    arch            full model config, so the architecture can be reconstructed
    epoch           for resuming at the right point
    state_dict      the weights (NOT the pickled module -- a state_dict is just an
                    ordered dict of tensors, portable across refactors, whereas a
                    pickled nn.Module couples the file to the source layout)
    optimizer       AdamW's exp_avg / exp_avg_sq per parameter. Dropping these is the
                    classic mistake: resuming reinitialises the moment estimates to
                    zero, which produces a visible loss spike on the first resumed
                    epoch. tests/test_checkpoint.py asserts there is no such spike.
    scheduler       cosine annealing tracks its own step count
    monitor_best    so a resume does not forget the best score and overwrite best.pt
                    with something worse
    config          the whole run config, snapshotted
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Any | None, epoch: int, monitor_best: float,
                    config: dict, arch: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": arch,                       # full dict, not just type(model).__name__
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "monitor_best": monitor_best,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, *, model: nn.Module | None = None,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any | None = None,
                    map_location: str | torch.device = "cpu",
                    strict_arch: bool = True) -> dict:
    """Load a checkpoint, optionally restoring model/optimizer/scheduler in place.

    Args:
        strict_arch: raise if the checkpoint's arch dict does not match the model's.
            On by default -- silently loading arm C's weights into arm B would corrupt
            the ablation in a way no metric would reveal.
    """
    # weights_only=False: our dict holds the config too, not just tensors.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        if strict_arch and hasattr(model, "head_type"):
            want, got = ckpt["arch"].get("head", {}).get("type"), model.head_type
            if want is not None and want != got:
                raise ValueError(
                    f"architecture mismatch: checkpoint was trained with head "
                    f"{want!r}, this model has {got!r}. Loading it would silently "
                    f"mix ablation arms. Pass strict_arch=False to override."
                )
        model.load_state_dict(ckpt["state_dict"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt
