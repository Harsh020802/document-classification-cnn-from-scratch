"""Device selection.

Why this exists as its own module rather than a one-liner at the top of train.py:
every script (baselines, training, ablation, inference) needs the same answer, and
hard-coding `torch.device("mps")` anywhere would break the Kaggle fallback we
planned as insurance against an unsupported MPS op.

Priority order is mps -> cuda -> cpu. Note this differs from most tutorials
(including both reference repos), which check only `torch.cuda.is_available()` and
would silently return CPU on this machine.
"""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    Args:
        prefer: force a specific backend ("mps" / "cuda" / "cpu"). Used by the
            determinism check, where we may want to pin CPU to get bit-reproducible
            results for comparison against MPS.

    Returns:
        The selected torch.device.
    """
    if prefer is not None:
        return torch.device(prefer)

    # MPS first: on Apple Silicon it is the only accelerator available.
    # is_built() tells us the wheel has MPS compiled in; is_available() tells us
    # the hardware and macOS version actually support it. Both must hold.
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def device_report(device: torch.device) -> str:
    """One-line human-readable summary, printed at the start of every run.

    Worth logging because a silent fallback to CPU is the most common cause of
    "why is this epoch taking 10 minutes" — better to see the device up front.
    """
    if device.type == "mps":
        return "mps (Apple Silicon GPU via Metal)"
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"cuda ({name}, {vram:.1f} GB VRAM)"
    return f"cpu ({torch.get_num_threads()} threads)"
