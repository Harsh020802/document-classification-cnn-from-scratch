"""Metrics.

Macro-F1 is the primary metric, NOT accuracy. With a 5.17x imbalance a model can ignore
Resume entirely (120 of 3,482) and still post a respectable accuracy. Macro-F1 weights
all ten classes equally, so an ignored class drags it down visibly.

sklearn is used for the metric computation itself (allowed: metrics only), but the
running accumulation is hand-written -- it is three lines and importing a framework for
it would obscure what is being averaged.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


class RunningAverage:
    """Accumulates per-batch values into an epoch average.

    Deliberately lighter than victoresque's MetricTracker, which holds a pandas
    DataFrame per tracker and uses chained indexing that newer pandas warns on. A dict
    of floats does this better.

    IMPORTANT on weighting: `n` is the batch size, so the average is per-SAMPLE, not
    per-batch. With a final partial batch the two differ, and the per-batch version
    silently over-weights the short one.
    """

    def __init__(self) -> None:
        self._total: dict[str, float] = {}
        self._count: dict[str, int] = {}

    def update(self, key: str, value: float, n: int = 1) -> None:
        self._total[key] = self._total.get(key, 0.0) + value * n
        self._count[key] = self._count.get(key, 0) + n

    def avg(self, key: str) -> float:
        c = self._count.get(key, 0)
        return self._total[key] / c if c else 0.0

    def result(self) -> dict[str, float]:
        return {k: self.avg(k) for k in self._total}

    def reset(self) -> None:
        self._total.clear()
        self._count.clear()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction correct. Computed under no_grad -- this is a report, not a loss."""
    with torch.no_grad():
        return (logits.argmax(dim=1) == targets).float().mean().item()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> float:
    """Unweighted mean of per-class F1.

    zero_division=0: if a class is never predicted, its precision is undefined. Scoring
    that as 0 rather than raising is correct here -- a model that never predicts Resume
    should be penalised, not crash.
    """
    return float(f1_score(y_true, y_pred, average="macro",
                          labels=list(range(n_classes)), zero_division=0))


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray,
                     class_names: list[str]) -> dict[str, dict[str, float]]:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    return {n: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])}
            for i, n in enumerate(class_names)}


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
